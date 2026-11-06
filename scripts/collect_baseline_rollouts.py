#!/usr/bin/env python
"""Phase 6.2 - collect baseline rollouts from the frozen 1.5B control arm.

This drives `Qwen/Qwen2.5-Coder-1.5B-Instruct` (the *base* model, no adapter) through
the project's own environment, one episode at a time, and writes a v3 trace row per
rollout. It is the only script that produces the Phase 6 numbers, so it is deliberately
boring: it decides nothing the config does not already say.

Design rules, each of which exists because the alternative produces a misleading number:

1. **The config is the only source of settings.** Every value read here comes from
   `configs/eval/baseline_phase6.yaml`. Nothing is defaulted in code, because a default
   in code is a setting that the config claims to have fixed and did not.

2. **Grading is the project's verifier, never a re-implementation.** `session.grade()`
   calls `verify_sandbox`, the same function the RL reward path calls
   (AGENT_BUILD_SPEC 5.6). The reward the baseline reports and the reward GRPO trains on
   are therefore the same number by construction.

3. **Every trajectory class is saved.** Success, failure, gated, and tool-error rollouts
   all get a row. A batch that keeps only the successes is not a baseline; it is a
   best-of-n, and its mean reward is a fiction. Crashed rollouts also get a row (with
   `termination_reason: "error"`), because a vanished rollout silently removes a zero
   from the denominator.

4. **A zero-variance batch is refused.** 6.2 explicitly asks to avoid identical rewards
   across all rollouts. If every rollout scores the same, the batch is reported as a
   failure of the *harness*, not as a baseline result - a constant reward gives GRPO no
   gradient, so this is also the signal that Phase 7 cannot start.

5. **The held-out split is unreachable from here.** The split is read from the config
   and asserted to be `train`. `--split` is not a flag, so this cannot be flipped by
   accident on a command line.

Usage
-----
    # the whole frozen subset, 8 rollouts per task
    python scripts/collect_baseline_rollouts.py

    # a quick smoke: 1 task, 2 rollouts, no variance requirement
    python scripts/collect_baseline_rollouts.py --tasks 1 --rollouts 2 --allow-uniform

    # dry run: check the server and the task list, collect nothing
    python scripts/collect_baseline_rollouts.py --dry-run

Every command that can touch a GPU must carry `CUDA_VISIBLE_DEVICES=0`
(docs/PHASE5_GPU_RESOURCE_RULES.md rules 1-2). This script does not set it for you: a
script that quietly overrides the environment makes the rule unenforceable.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import signal
import stat
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.context_budget import (  # noqa: E402
    STATUS_EXCEEDED,
    STATUS_OK,
    STATUS_TOKENIZER_UNAVAILABLE,
    CompactionPolicy,
    ContextBudget,
    Probe,
    TokenizerUnavailableError,
    compact_messages,
    infrastructure_failure,
    is_infrastructure_failure,
    load_tokenizer,
    preflight,
    reward_eligible_rows,
)
from analysis.sampling_seeds import (  # noqa: E402
    SAMPLING_SEED_DERIVATION,
    SeedDerivationError,
    assert_unique_seeds,
    derive_rollout_seed,
    derive_turn_seed,
    plan_digest,
    plan_seeds,
)
from analysis.trace import (  # noqa: E402
    SCHEMA_VERSION,
    Timer,
    build_row,
    collect_patch,
    secure_write_private_log,
    write_jsonl,
)
from envs.coding_env.session import EpisodeSession, find_task  # noqa: E402
from envs.coding_env.tools import TOOL_DEFINITIONS  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "eval" / "baseline_phase6.yaml"

log = logging.getLogger("phase6.collect")


# --------------------------------------------------------------------------- errors


class CollectorError(RuntimeError):
    """A precondition the collector refuses to run without."""


class SeedPlanError(CollectorError):
    """The planned sampling seeds are unusable, so the batch would not be independent.

    `analysis.sampling_seeds.SeedDerivationError` is a `ValueError` about the inputs; this is
    the collector's own error about the *batch*. They are kept separate so a caller can tell
    "you passed a bad seed to the derivation" from "this run cannot produce independent
    rollouts", and so the CLI reports the second with a config-level message rather than a
    traceback from inside a pure function.
    """


class ResumeFileError(CollectorError):
    """A persisted rollouts file cannot be safely resumed from.

    Raised for every resume condition that is not a clean interrupted append: a malformed
    line that is not the final one, a duplicate `rollout_id`, an unidentifiable trailing
    fragment, or rows belonging to another run. Resuming from such a file would mean
    guessing which evidence was real, and a guessed batch is not reproducible - so the
    collector refuses and asks for a new `run_id` instead.
    """


class UniformRewardError(CollectorError):
    """Every rollout scored the same, so the batch measures nothing (6.2)."""


class InfrastructureNotReadyError(CollectorError):
    """The run cannot produce reward-eligible evidence and must not proceed (6B).

    Raised before collection when the tokenizer is unavailable, and after collection
    when a guard shows the batch is not a baseline. Distinguished from
    `UniformRewardError` because the cause is the environment, not the model.
    """

    def __init__(self, message: str, *, report: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.report = report or {}


# -------------------------------------------------------------------------- config


@dataclass(frozen=True)
class BaselineConfig:
    """The frozen 6.1 settings, read once and used without defaults."""

    raw: dict[str, Any]

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> BaselineConfig:
        with Path(path).open(encoding="utf-8") as fh:
            return cls(raw=yaml.safe_load(fh) or {})

    def _get(self, dotted: str) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                raise CollectorError(
                    f"{CONFIG_PATH.name} is missing '{dotted}'. The baseline is frozen: a "
                    "missing key is a broken config, not a value to default."
                )
            node = node[part]
        return node

    # model
    @property
    def model_name(self) -> str:
        return self._get("model.name")

    @property
    def model_revision(self) -> str:
        return self._get("model.revision")

    # server
    @property
    def base_url(self) -> str:
        return str(self._get("server.base_url"))

    @property
    def api_key(self) -> str:
        return str(self._get("server.api_key"))

    @property
    def served_model_name(self) -> str:
        return str(self._get("server.served_model_name"))

    # output
    def output_path(self, key: str) -> Path:
        return REPO_ROOT / str(self._get(f"output.{key}"))

    @property
    def adapter(self) -> str | None:
        value = self._get("model.adapter")
        if value is not None:
            raise CollectorError(
                "model.adapter is set, but this is the *baseline* arm: a loaded adapter "
                "makes every number a post-RL number under a pre-RL label."
            )
        return None

    # sampling
    @property
    def temperature(self) -> float:
        return float(self._get("sampling.temperature"))

    @property
    def top_p(self) -> float:
        return float(self._get("sampling.top_p"))

    @property
    def seed(self) -> int:
        return int(self._get("sampling.seed"))

    @property
    def seed_policy(self) -> str:
        """How `sampling.seed` maps onto individual model requests.

        `per_rollout` (the default, and the only value Phase 6H plans) derives one seed per
        `(task_id, rollout_index)` and sends it on every turn of that rollout.
        `per_turn` derives a further seed per turn from the rollout seed.
        `literal` reproduces the pre-6H behaviour of sending `sampling.seed` verbatim on
        every request; it is retained only so a historical config is still interpretable,
        and a run using it can never be called a multi-rollout baseline.
        """
        value = self._get("sampling.seed_policy")
        allowed = {"per_rollout", "per_turn", "literal"}
        if value not in allowed:
            raise CollectorError(f"sampling.seed_policy is {value!r}; expected one of {sorted(allowed)}")
        return str(value)

    @property
    def seed_derivation(self) -> str:
        """The derivation label this run intends to use.

        Checked against `analysis.sampling_seeds.SAMPLING_SEED_DERIVATION` before the first
        model call. A config naming a different algorithm would produce seeds the recorded
        label cannot explain, so the mismatch is refused rather than recorded.
        """
        value = self._get("sampling.seed_derivation")
        if value != SAMPLING_SEED_DERIVATION:
            raise CollectorError(
                f"sampling.seed_derivation is {value!r} but this code derives seeds as "
                f"{SAMPLING_SEED_DERIVATION!r}. Re-deriving under a different algorithm would "
                "silently change every rollout's sample; update the config deliberately."
            )
        return str(value)

    @property
    def max_tokens_per_turn(self) -> int:
        return int(self._get("sampling.max_tokens_per_turn"))

    # episode
    @property
    def timeout_s_per_rollout(self) -> int:
        return int(self._get("episode.timeout_s_per_rollout"))

    @property
    def transport(self) -> str:
        return str(self._get("episode.transport"))

    # prompt
    @property
    def system_message(self) -> str:
        return str(self._get("prompt.system_message"))

    @property
    def prompt_version(self) -> str:
        return str(self._get("prompt.version"))

    @property
    def tools_version(self) -> str:
        return str(self._get("tools.version"))

    # run
    @property
    def run_id(self) -> str:
        return str(self._get("output.run_id"))

    @property
    def split(self) -> str:
        split = str(self._get("tasks.split"))
        if split != "train":
            raise CollectorError(
                f"tasks.split is {split!r}; Phase 6 runs on train only. The held-out split "
                "is sealed and run exactly once, at the end of the project "
                "(docs/heldout_seal.md)."
            )
        return split

    @property
    def task_subset(self) -> list[str]:
        return list(self._get("run.task_subset"))

    @property
    def rollouts_per_task(self) -> int:
        return int(self._get("run.rollouts_per_task"))

    @property
    def require_reward_variance(self) -> bool:
        return bool(self._get("run.require_reward_variance"))

    @property
    def min_distinct_rewards(self) -> int:
        return int(self._get("run.min_distinct_rewards"))

    # context budget (Phase 6B). These have no defaults on purpose: the safety margin and
    # the tokenizer path decide whether a request is sent at all, so they are recorded in
    # the config rather than inferred from whatever the environment happens to contain.
    @property
    def max_model_len(self) -> int:
        return int(self._get("context.max_model_len"))

    @property
    def safety_margin_tokens(self) -> int:
        return int(self._get("context.safety_margin_tokens"))

    @property
    def tokenizer_path(self) -> Path:
        return Path(str(self._get("context.tokenizer_path"))).expanduser()

    @property
    def observation_char_cap(self) -> int:
        return int(self._get("context.observation_char_cap"))

    @property
    def keep_recent_tool_messages(self) -> int:
        return int(self._get("context.keep_recent_tool_messages"))

    @property
    def min_valid_rollouts(self) -> int:
        return int(self._get("run.min_valid_rollouts"))

    @property
    def diagnostic_only(self) -> bool:
        """When true the run may be uniform-variance, but is never training evidence."""
        if "run.diagnostic_only" not in _flatten_keys(self.raw):
            return False
        return bool(self._get("run.diagnostic_only"))

    def context_budget(self) -> ContextBudget:
        return ContextBudget(
            max_model_len=self.max_model_len,
            reserved_completion_tokens=self.max_tokens_per_turn,
            safety_margin_tokens=self.safety_margin_tokens,
        )


def _flatten_keys(node: Any, prefix: str = "") -> set[str]:
    """Dotted key set, used to tell an absent optional key from a falsey one."""
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            dotted = f"{prefix}{key}"
            keys.add(dotted)
            keys |= _flatten_keys(value, f"{dotted}.")
    return keys


#: Where each CLI override lives in the config document. The override is applied to the
#: raw mapping rather than to an attribute, because every setting on `BaselineConfig` is a
#: read-only view over `raw` - there is nothing to assign to.
_OVERRIDE_PATHS: dict[str, tuple[str, ...]] = {
    "diagnostic_only": ("run", "diagnostic_only"),
    "tokenizer_path": ("context", "tokenizer_path"),
}


def _with_overrides(cfg: BaselineConfig, **overrides: Any) -> BaselineConfig:
    """A new frozen config with CLI overrides applied, leaving the original untouched.

    `dataclasses.replace` on this dataclass only replaces `raw`, so an override is written
    into a deep copy of the raw document and a fresh instance is built around it. The input
    is never mutated: a caller that holds the config it loaded must not find it changed
    underneath it, and the frozen dataclass is what enforces that.
    """
    raw = copy.deepcopy(cfg.raw)
    for name, value in overrides.items():
        try:
            *parents, leaf = _OVERRIDE_PATHS[name]
        except KeyError as exc:
            raise CollectorError(f"no config path is registered for override {name!r}") from exc
        node = raw
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return replace(cfg, raw=raw)


# ------------------------------------------------------------------ model client


def _client(base_url: str, api_key: str):
    """An OpenAI client pointed at the local vLLM server."""
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover - dependency is present
        raise CollectorError(f"the `openai` package is required to drive the model: {e}") from e
    return OpenAI(base_url=base_url, api_key=api_key, timeout=300.0)


def check_server(cfg: BaselineConfig) -> str:
    """Confirm the server is up and serving the model the config pins.

    A reachable server answering with a *different* model would produce numbers that
    look like a 1.5B baseline and are not one, so the model id is compared, not just the
    status code.
    """
    client = _client(cfg.base_url, cfg.api_key)
    try:
        served = [m.id for m in client.models.list().data]
    except Exception as e:
        raise CollectorError(
            f"cannot reach the model server at {cfg.base_url}: {e}\n"
            "Start it with CUDA_VISIBLE_DEVICES=0 and the pinned revision; see "
            "docs/PHASE6_BASELINE_EVALUATION.md."
        ) from e
    if cfg.served_model_name not in served:
        raise CollectorError(f"server at {cfg.base_url} serves {served}, but the config pins {cfg.served_model_name!r}")
    return cfg.served_model_name


# --------------------------------------------------------------------- tool calling
#
# vLLM 0.6.1 is started without `--enable-auto-tool-choice`, so the model emits tool calls
# as JSON in the message content rather than as `tool_calls`. The parser below accepts both
# shapes plus an inline fenced block, because a 1.5B model will use whichever it prefers and
# a parser that only understands one of them records "the model called no tools" for a
# rollout that called several - a false statement about the trajectory.

_KNOWN_TOOLS = {d["name"] for d in TOOL_DEFINITIONS}


def _as_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def parse_tool_calls(message: Any) -> list[tuple[str, dict[str, Any]]]:
    """Extract `(name, arguments)` pairs from one assistant message.

    Accepted shapes, in order of preference: native `tool_calls`, a whole-message JSON
    object, a fenced ```json block, and a bare array of call objects. Anything else is
    a reasoning-only turn and returns `[]` - which is not an error, because a turn with no
    tool call is a legal turn the agent loop counts against the turn budget.
    """
    native = getattr(message, "tool_calls", None) or []
    out: list[tuple[str, dict[str, Any]]] = []
    for call in native:
        fn = getattr(call, "function", None)
        if fn is not None:
            name = getattr(fn, "name", "")
            if name in _KNOWN_TOOLS:
                out.append((name, _as_arguments(getattr(fn, "arguments", None))))
    if out:
        return out

    content = getattr(message, "content", None) or ""
    if not isinstance(content, str) or not content.strip():
        return []

    for candidate in _json_candidates(content):
        parsed = _call_from_payload(candidate)
        if parsed:
            return parsed
    return []


def _json_candidates(text: str) -> list[Any]:
    """Every JSON value that might be a tool call in `text`, best first."""
    found: list[Any] = []
    stripped = text.strip()
    try:
        found.append(json.loads(stripped))
    except json.JSONDecodeError:
        pass

    # Fenced blocks: ```json { ... } ```
    marker = "```"
    if marker in text:
        for chunk in text.split(marker)[1::2]:
            body = chunk
            if body.lstrip().lower().startswith("json"):
                body = body.lstrip()[4:]
            try:
                found.append(json.loads(body.strip()))
            except json.JSONDecodeError:
                continue

    # A JSON object embedded in prose: the outermost {...} that parses.
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        try:
            found.append(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass
    return found


def _call_from_payload(payload: Any) -> list[tuple[str, dict[str, Any]]]:
    """Normalise one parsed payload into tool calls, or `[]` if it is not one."""
    items: list[Any]
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        if "tool_calls" in payload:
            items = payload["tool_calls"]
        elif "name" in payload:
            items = [payload]
        else:
            # `{"list_files": {"path": "."}}` - the tool name used as the key. Not the
            # documented format, but a predictable misreading of it that the 1.5B model
            # produced, so it is accepted rather than scored as a no-op turn. Only a
            # mapping with exactly one known tool name qualifies.
            named = [(k, v) for k, v in payload.items() if k in _KNOWN_TOOLS]
            if len(named) != 1:
                return []
            name, raw = named[0]
            return [(name, _as_arguments(raw))]
    else:
        return []

    out: list[tuple[str, dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name")
        if name in _KNOWN_TOOLS:
            args = fn.get("arguments", fn.get("parameters", {}))
            out.append((name, _as_arguments(args)))
    return out


# ------------------------------------------------------------------- one rollout


def _is_context_length_error(exc: BaseException) -> bool:
    """True when the server rejected a request for exceeding the context window.

    Matched on the server's own wording rather than on an OpenAI exception class, because
    vLLM raises `BadRequestError` for several unrelated causes and only this one means the
    model never executed.
    """
    text = str(exc)
    return "maximum context length" in text or "context length is" in text


@dataclass
class RolloutOutcome:
    """What one episode produced, kept separate from the trace row."""

    row: dict[str, Any]
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stopped_on: str = "submitted"
    transcript_chars: int = 0
    #: Set when the attempt never became a reward-eligible episode. The record is stored
    #: alongside the row so the batch can count it without re-parsing messages.
    infrastructure_failure: dict[str, Any] | None = None
    #: Context-budget telemetry for the first request of this rollout, in the public
    #: (schema-declared, counts-only) shape. The same record is written to the private
    #: sidecar, so the two files describe one measurement rather than two.
    context_budget: dict[str, Any] | None = None
    compaction_events: int = 0

    @property
    def reward_eligible(self) -> bool:
        return self.infrastructure_failure is None


def _system_prompt(cfg: BaselineConfig) -> str:
    """The configured system message with the tool catalogue appended as text.

    The catalogue is rendered from `TOOL_DEFINITIONS` rather than typed out here so the
    prompt cannot drift from the tools the sandbox will actually accept.
    """
    lines = [cfg.system_message, "", "## Tools", ""]
    lines.append(
        "To call a tool, reply with exactly one JSON object and nothing else:\n"
        '  {"name": "<tool>", "arguments": { ... }}\n'
        "Do not use the OpenAI `tool_calls` shape. Do not wrap the JSON in prose. "
        "Put every argument inside `arguments`."
    )
    lines.append("")
    lines.append("Example - read the first 50 lines of a file:")
    lines.append('  {"name": "read_file", "arguments": {"path": "src/app.py", "start": 1, "end": 50}}')
    lines.append("")
    lines.append("Available tools:")
    for definition in TOOL_DEFINITIONS:
        params = definition.get("parameters", {})
        props = params.get("properties", {}) or {}
        required = set(params.get("required", []) or [])
        arg_desc = ", ".join(
            f"{name}: {spec.get('type', 'any')}" + ("" if name in required else " (optional)")
            for name, spec in props.items()
        )
        lines.append(f"- {definition['name']}({arg_desc}): {definition.get('description', '').strip()}")
    return "\n".join(lines)


def _task_instruction(task: Any) -> str:
    """The task's `instruction:` text, verbatim and unmodified.

    A missing or empty instruction is an error rather than an empty user turn: an episode
    started with no prompt measures the harness, not the model, and would look like a
    legitimate rollout scored 0.0.
    """
    raw = (task.data or {}).get("instruction")
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise CollectorError(f"{task.id}: task.yaml has no `instruction:` - refusing to start a blank episode")
    return text


def _request_seed(
    cfg: BaselineConfig,
    *,
    task_id: str,
    rollout_index: int,
    turn_index: int | None = None,
) -> int:
    """The seed to send on a model request, per `sampling.seed_policy`.

    This is the single place the request seed is decided, so there is exactly one answer to
    "what seed did that request carry?" and it is the same answer the row records.

    * `per_rollout` - one derived seed per rollout, repeated on every turn. The serving API
      `seed` field is request-level, so a rollout is only reproducible as a whole if each of
      its requests carries the same seed.
    * `per_turn` - `turn_index` must be supplied; the per-turn seed is derived from the
      rollout seed, not from the base seed, so a rollout stays a coherent unit.
    * `literal` - `cfg.seed` verbatim. Reproduces pre-6H behaviour, in which every rollout
      of every task is the same sample; refused for any plan with more than one rollout
      (see `_assert_seed_policy_usable`).
    """
    policy = cfg.seed_policy
    if policy == "literal":
        return cfg.seed
    rollout_seed = derive_rollout_seed(cfg.seed, task_id, rollout_index)
    if policy == "per_rollout":
        return rollout_seed
    if turn_index is None:
        raise CollectorError("sampling.seed_policy is 'per_turn' but no turn_index was supplied")
    return derive_turn_seed(rollout_seed, turn_index)


def _assert_seed_policy_usable(cfg: BaselineConfig, *, n_rollouts: int) -> None:
    """Refuse a seed policy that cannot yield independent rollouts.

    A `literal` policy sends one seed on every request, so a "multi-rollout baseline" under
    it is one sample measured `n_rollouts` times. Phase 6G did exactly that, and its uniform
    -1.0 cannot be read as a statement about the model for that reason: one observation
    repeated four times is not a sample of four. The model may still score -1.0 on all four
    independent samples. The guard is here rather than in docs because the failure is silent:
    nothing about the artifacts looks wrong.
    """
    if cfg.seed_policy == "literal" and n_rollouts > 1:
        raise CollectorError(
            f"sampling.seed_policy is 'literal' with {n_rollouts} rollouts per task: every rollout "
            "would send the same seed and be the same sample. Set the policy to 'per_rollout' to "
            "derive an independent seed per rollout."
        )


def _rollouts_path(cfg: BaselineConfig) -> Path:
    """Where this run's public rows are written.

    Deliberately `cfg.output_path("rollouts")` and not a path assembled from a run-id
    directory: the `output.*` keys already name the full path, so building `output_dir /
    run_id / "rollouts.jsonl"` here would disagree with `_persist_evidence` and `--resume`
    would silently find nothing and re-sample the whole batch.
    """
    return cfg.output_path("rollouts")


def _truncate_to_last_newline(path: Path) -> int:
    """Cut `path` back to its last complete line, flush it and fsync it.

    Returns the offset the file was truncated to, which equals its new size.

    Resuming with a dangling partial line in place is not enough. The collector later
    *rewrites* `rollouts.jsonl` from the in-memory rows, so the fragment would disappear
    anyway - but only after `_persist_evidence`. If the resumed collector reads a fragment,
    re-samples the missing rollout and then dies before persisting, the file still holds the
    fragment, and the next resume has to re-derive the same conclusion from a file that has
    been silently growing tail garbage. Repairing the file once, at the moment the fragment
    is identified, leaves every subsequent attempt with a well-formed JSONL to reason about.

    Truncation is to a byte offset and not a rewrite because the bytes before that offset
    were already fsynced; rewriting them would risk damaging evidence to repair a tail.
    """
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return len(data)
    cut = data.rfind(b"\n") + 1  # 0 when there is no newline at all: the whole file goes
    with path.open("r+b") as handle:
        handle.truncate(cut)
        handle.flush()
        os.fsync(handle.fileno())
    # fsync the directory too: on a crash the truncate itself must be durable, not just the
    # data, or a resume could observe the old size with the new content.
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
    return cut


def _load_persisted(cfg: BaselineConfig) -> dict[str, dict[str, Any]]:
    """Rows a previous attempt of this run already persisted, keyed by rollout_id.

    Only `rollouts.jsonl` is consulted. A row is the durable record of a finished episode;
    the private log is written alongside it but is not the source of truth for resume.

    The loader is strict on purpose. One malformed shape *is* recoverable - an interrupted
    append, which leaves a partial object after the last newline with no trailing newline -
    and it is repaired here by truncating the file to that boundary. Everything else is
    refused, because each of the alternatives silently discards or corrupts evidence:

    * a malformed line followed by more content is not an interrupted append. The file was
      written by something else, or edited; dropping that line would re-sample a rollout
      whose row may still be referenced elsewhere, and keep rows of unknown provenance.
    * a duplicate `rollout_id` means two different rows claim the same episode. Last-wins
      would let a later line override the seed and reward the earlier one recorded.
    * rows whose `run_id` differs are another batch's evidence. Dropping them silently is
      how a mixed directory becomes a "resumed" run with fewer rollouts than it reports.
    * a trailing fragment that cannot be identified as a partial row (no `{`, or it parses
      but is not an object) is not something truncation should be applied to blind.
    """
    path = _rollouts_path(cfg)
    if not path.exists():
        return {}

    raw = path.read_bytes()
    if not raw:
        return {}

    # Split on the newline byte, keeping track of whether the file ends with one. The last
    # element after `split` is either `b""` (clean file) or the unterminated tail.
    pieces = raw.split(b"\n")
    trailing = pieces.pop()  # b"" when the file ended with a newline
    repaired_tail = False
    if trailing:
        # No trailing newline: the final element is an interrupted append, or corruption.
        if not trailing.lstrip().startswith(b"{"):
            raise ResumeFileError(
                f"{path} ends with an unterminated fragment that does not begin a JSON object; "
                "this is corruption, not an interrupted append. Start a new run_id."
            )
        try:
            json.loads(trailing)
        except json.JSONDecodeError:
            repaired_tail = True
        else:
            # It parses, so it is a complete row that merely lacks the newline. Keep it and
            # terminate the line rather than discarding a finished rollout.
            pieces.append(trailing)
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            log.info("resume: terminated a final row in %s that lacked a trailing newline", path)

    if repaired_tail:
        offset = _truncate_to_last_newline(path)
        log.warning(
            "resume: discarded an interrupted append in %s and truncated the file to %d byte(s); "
            "that rollout will be re-sampled",
            path,
            offset,
        )

    persisted: dict[str, dict[str, Any]] = {}
    for lineno, chunk in enumerate(pieces, start=1):
        if not chunk.strip():
            continue
        try:
            row = json.loads(chunk)
        except json.JSONDecodeError as exc:
            # Every remaining candidate is terminated, so this is a malformed line in the
            # middle of the file: not recoverable, and not the interrupted-append shape.
            raise ResumeFileError(
                f"{path} line {lineno} is not valid JSON ({exc.msg}); a malformed row before the end of "
                "the file is corruption, not an interrupted append. Start a new run_id."
            ) from exc
        if not isinstance(row, dict):
            raise ResumeFileError(f"{path} line {lineno} is not a JSON object")
        run_id = row.get("run_id")
        if run_id != cfg.run_id:
            raise ResumeFileError(
                f"{path} line {lineno} belongs to run_id {run_id!r}, not {cfg.run_id!r}. "
                "A run_id identifies one batch, so rows from another run are a layout mistake; "
                "point --resume at this run's own directory or start a new run_id."
            )
        rollout_id = row.get("rollout_id")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise ResumeFileError(f"{path} line {lineno} has no usable rollout_id")
        if rollout_id in persisted:
            raise ResumeFileError(
                f"{path} contains rollout_id {rollout_id!r} more than once. Keeping either copy would "
                "mean guessing which one is the recorded episode; start a new run_id."
            )
        persisted[rollout_id] = row
    return persisted


def run_one_rollout(
    *,
    cfg: BaselineConfig,
    client: Any,
    task: Any,
    rollout_id: str,
    rollout_index: int,
    deadline: float,
    tokenizer: Any = None,
) -> RolloutOutcome:
    """Drive one episode to completion and return its trace row.

    The loop is deliberately simple: build messages, measure them, ask the model, dispatch
    whatever tool calls it named, repeat until it submits, spends its tool budget, spends
    its turn budget, or runs out of wall clock. Each of those endings is recorded, because
    "the agent ran out of turns" and "the agent gave up" are different facts about the
    model and only the harness can tell them apart.

    `tokenizer` is required for exact context budgeting. When it is None the first turn
    produces an `infrastructure_error` row rather than a model call: an unmeasurable
    prompt cannot be sent safely, and guessing would recreate the Phase 6 bring-up 400s.

    `rollout_index` used to be accepted and ignored, which is how Phase 6G produced four
    identical rollouts from one seed. It now feeds `derive_rollout_seed`, so the sampling
    seed is a function of `(cfg.seed, task.id, rollout_index)` and every rollout samples
    differently while remaining reproducible from its own row.
    """
    # Derived once, before the first model call, so the value recorded in the row is by
    # construction the value the requests used.
    sampling_seed = _request_seed(cfg, task_id=task.id, rollout_index=rollout_index)

    timer = Timer.start()
    session = EpisodeSession.open(task)
    max_turns = session.max_turns

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(cfg)},
        # The user turn is the task's own `instruction:` field, verbatim. `Task` exposes no
        # `instruction` property, so it is read from `task.data` - and the prompt hash in
        # the trace records what was actually sent.
        {"role": "user", "content": _task_instruction(task)},
    ]
    # The tool catalogue goes into the system prompt, not into the request's `tools`
    # field. This server was started without --enable-auto-tool-choice, so sending
    # `tools` is a hard 400; the model is asked to emit tool calls as JSON instead and
    # `parse_tool_calls` reads them back. Keeping the catalogue in the prompt also means
    # the trace's prompt hash covers the tools the model actually saw.

    tokens = {"prompt": 0, "completion": 0, "total": 0}
    step_token_counts: list[int] = []
    turns_used = 0
    stopped_on = "max_turns"
    error: BaseException | None = None
    result = None
    private_tool_log: list[dict[str, Any]] = []
    compaction_events = 0
    first_probe: Probe | None = None
    infra: dict[str, Any] | None = None

    budget = cfg.context_budget()
    compaction_policy = CompactionPolicy(
        observation_char_cap=cfg.observation_char_cap,
        keep_recent_tool_messages=cfg.keep_recent_tool_messages,
    )

    try:
        while True:
            if turns_used >= max_turns:
                stopped_on = "max_turns"
                break
            if time.monotonic() > deadline:
                stopped_on = "timeout"
                break
            if session.done:
                stopped_on = "max_tool_calls" if not session.sandbox.toolbox.submitted else "submitted"
                break

            # ---- context preflight, before the request leaves this process ----------
            # The Phase 6 bring-up sent requests the server rejected with a 400, which
            # then looked like model failures. Measuring first turns that into a decision
            # this harness makes and records, instead of an error it discovers.
            probe = preflight(messages, tokenizer, budget)
            if turns_used == 0:
                first_probe = probe

            if probe.status == STATUS_TOKENIZER_UNAVAILABLE:
                infra = infrastructure_failure(
                    error_type="tokenizer_unavailable",
                    stage="preflight",
                    message=probe.detail or "tokenizer unavailable",
                    probe=probe,
                )
                stopped_on = "infrastructure_error"
                break

            if probe.status == STATUS_EXCEEDED and turns_used == 0:
                # The initial prompt alone is too large: do not call the model at all.
                infra = infrastructure_failure(
                    error_type="context_budget_exceeded",
                    stage="preflight",
                    message=probe.detail,
                    probe=probe,
                )
                stopped_on = "infrastructure_error"
                break

            if probe.status == STATUS_EXCEEDED:
                # A later turn outgrew the window. Compact deterministically, re-measure,
                # and only give up if compaction cannot make room.
                compaction = compact_messages(messages, policy=compaction_policy, tokenizer=tokenizer, budget=budget)
                if compaction.compacted:
                    messages = compaction.messages
                    compaction_events += 1
                    probe = preflight(messages, tokenizer, budget)
                if probe.status != STATUS_OK:
                    infra = infrastructure_failure(
                        error_type="context_budget_exceeded",
                        stage="preflight_after_compaction",
                        message=probe.detail,
                        probe=probe,
                    )
                    stopped_on = "infrastructure_error"
                    break

            turns_used += 1
            try:
                response = client.chat.completions.create(
                    model=cfg.served_model_name,
                    messages=messages,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    max_tokens=cfg.max_tokens_per_turn,
                    seed=(
                        sampling_seed
                        if cfg.seed_policy == "per_rollout"
                        else _request_seed(cfg, task_id=task.id, rollout_index=rollout_index, turn_index=turns_used)
                    ),
                    timeout=max(30.0, deadline - time.monotonic()),
                )
            except Exception as e:  # the server is a dependency; its failure is recorded
                error = e
                # A context-length 400 that slipped past preflight is still an
                # infrastructure fault, not a model failure: the model never ran.
                if _is_context_length_error(e):
                    infra = infrastructure_failure(
                        error_type="context_budget_exceeded",
                        stage="server",
                        message=str(e),
                        probe=first_probe,
                    )
                    stopped_on = "infrastructure_error"
                else:
                    stopped_on = "error"
                break

            usage = getattr(response, "usage", None)
            if usage is not None:
                tokens["prompt"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                tokens["completion"] += int(getattr(usage, "completion_tokens", 0) or 0)
                tokens["total"] = tokens["prompt"] + tokens["completion"]
            step_token_counts.append(int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else None)

            message = response.choices[0].message
            calls = parse_tool_calls(message)

            if not calls:
                # A turn with no tool call: keep the text in the transcript so the row is
                # auditable, and let the turn budget bound the loop.
                messages.append({"role": "assistant", "content": getattr(message, "content", "") or ""})
                continue

            # One id per emitted call, and the tool result below must reuse that same id
            # one-for-one. Duplicate or missing ids are a 400 from the server.
            emitted = [f"call_{turns_used}_{i}" for i in range(len(calls))]
            messages.append(
                {
                    "role": "assistant",
                    "content": getattr(message, "content", None),
                    "tool_calls": [
                        {
                            "id": emitted[i],
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                        for i, (name, args) in enumerate(calls)
                    ],
                }
            )

            for i, (name, args) in enumerate(calls):
                observed = session.tool(name, args)
                private_tool_log.append(
                    {"turn": turns_used, "tool": name, "arguments": args, "output": observed["output"]}
                )
                messages.append(
                    {
                        "role": "tool",
                        "content": str(observed["output"]),
                        "tool_call_id": emitted[i],
                    }
                )
                if observed.get("done"):
                    stopped_on = "submitted" if session.sandbox.toolbox.submitted else "max_tool_calls"
                    break
            if stopped_on in {"submitted", "max_tool_calls"}:
                break

        result = session.grade()
        # The verifier is the authority on how the episode ended; the loop only knows the
        # order the stopping conditions fired.
        if error is None:
            stopped_on = (
                "submitted"
                if result.submitted
                else "max_tool_calls"
                if session.sandbox.toolbox.n_calls >= session.sandbox.toolbox.max_tool_calls
                else stopped_on
            )
        patch = collect_patch(task, session.sandbox.repo_dir)
    except Exception as e:  # noqa: BLE001 - a crashed rollout must still produce a row
        error = e
        result = None
        stopped_on = "error"
        patch = ""
    finally:
        try:
            session.close()
        except Exception:  # noqa: BLE001 - cleanup failure must not mask the rollout
            pass

    row = build_row(
        run_id=cfg.run_id,
        rollout_id=rollout_id,
        task=task,
        result=result,
        tool_calls=session.sandbox.toolbox.log if result is not None or error is None else [],
        timer=timer,
        transport=cfg.transport,
        agent_kind="policy",
        model=cfg.model_name,
        # The baseline arm loads no adapter, so this is null by definition rather than by
        # omission - and the row says so.
        checkpoint_id=None,
        random_seed=sampling_seed,
        base_seed=cfg.seed,
        sampling_seed=sampling_seed,
        sampling_seed_derivation=cfg.seed_derivation,
        rollout_index=rollout_index,
        seed_policy=cfg.seed_policy,
        prompt_config_version=cfg.prompt_version,
        candidate_patch=patch,
        token_usage=tokens if tokens["total"] else None,
        step_token_counts=step_token_counts or None,
        turns_used=turns_used,
        error=error,
    )

    if error is not None:
        row["termination_reason"] = "error"
    elif infra is not None:
        # An infrastructure-invalid attempt is not a scored episode. The verifier is never
        # consulted, so there is no reward to report and none is invented: every numeric
        # result field becomes null rather than 0.0, because 0.0 is a measurement.
        row["termination_reason"] = "infrastructure_error"
        row["error"] = {
            "type": str(infra["error_type"]),
            "message": str(infra["message"]),
        }
        empty_result = dict.fromkeys(
            (
                "floor_tests",
                "pass_fraction",
                "reward",
                "submitted",
                "tests_passed",
                "tests_total",
            ),
            None,
        )
        empty_result["gates"] = []
        row["result"] = empty_result
        row["infrastructure_failure"] = infra
    elif row["termination_reason"] != stopped_on and stopped_on in {
        "timeout",
        "max_turns",
        "max_tool_calls",
    }:
        # `build_row` infers termination from the grade, which cannot see a wall-clock
        # timeout or a turn cap. The loop can, so the loop wins for those two.
        row["termination_reason"] = stopped_on

    # ---- context-budget telemetry, made public -------------------------------------
    # Phase 6E wrote this only into the private sidecar, but the run summary read
    # `row["context_budget"]` from the public row, so `summary.context_budget` was always
    # null - the telemetry existed and was silently unreachable at the level it was
    # reported. It now lives on the public row too.
    #
    # It is safe to publish because it is counts only: which fields appear is fixed by
    # `Probe.to_dict`, and no prompt text, tool argument or observation can enter through
    # it. The private sidecar keeps the same record for continuity with Phase 6E.
    row["context_budget"] = _public_context_telemetry(first_probe, compaction_events)

    return RolloutOutcome(
        row=row,
        tool_calls=private_tool_log,
        stopped_on=row["termination_reason"],
        transcript_chars=len(json.dumps(messages)),
        infrastructure_failure=infra,
        # Deliberately the same record as the public row: one measurement, written twice for
        # two audiences. `Probe.to_dict()` is not used here because its key set is not what
        # the schema accepts, and the private sidecar must stay joinable to the public row.
        context_budget=row["context_budget"],
        compaction_events=compaction_events,
    )


# ------------------------------------------------------------------ batch driver


def collect(
    cfg: BaselineConfig,
    *,
    tasks_limit: int | None = None,
    rollouts_override: int | None = None,
    allow_uniform: bool = False,
    tokenizer: Any = None,
    task_ids_override: list[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Collect the batch and return a summary. Raises `UniformRewardError` if required.

    Raises `InfrastructureNotReadyError` before the first request when no tokenizer is
    available, and after the loop when a run-level guard shows the batch is not evidence.

    `resume=True` keeps rollouts a previous attempt of the same `run_id` already persisted,
    and re-derives their seeds to prove the batch is still one derivation. A run that
    re-sampled everything on resume would silently discard GPU time; a run that stitched two
    different derivations together would silently produce an unreproducible batch. This does
    neither: it reuses matching rows and refuses mismatched ones.
    """
    if tokenizer is None:
        # Fail closed before touching the server: without exact token counts every request
        # is a guess, and a guess is what produced the Phase 6 bring-up 400s.
        tokenizer = load_tokenizer(cfg.tokenizer_path, revision=cfg.model_revision)

    # Resolve and validate task selection before any network call, so a typo fails fast
    # with a clear message instead of surfacing later as a confusing batch.
    task_ids = cfg.task_subset[:tasks_limit] if tasks_limit else list(cfg.task_subset)
    if task_ids_override:
        # Refuse an unknown id rather than silently collecting nothing: a typo that
        # yields an empty batch would look like a task that produced no evidence.
        unknown = [t for t in task_ids_override if t not in cfg.task_subset]
        if unknown:
            raise CollectorError(f"task id(s) not in this run's task_subset: {', '.join(unknown)}")
        task_ids = list(task_ids_override)
    per_task = rollouts_override or cfg.rollouts_per_task

    # Decide every rollout's sampling seed up front, before the server is contacted. Two
    # things are established here and nowhere else:
    #
    #   1. The plan is collision-free (`assert_unique_seeds`). A collision means two
    #      "independent" rollouts are one sample recorded twice - the Phase 6G failure - so
    #      it must be caught by arithmetic rather than discovered in the results.
    #   2. The policy can actually produce independence (`_assert_seed_policy_usable`). A
    #      `literal` policy with N>1 rollouts is the same bug written in the config.
    #
    # Both checks run before `check_server`, so a misconfigured run costs nothing.
    plan = plan_seeds(cfg.seed, task_ids, per_task)
    try:
        assert_unique_seeds(plan)
    except SeedDerivationError as e:
        # A collision is a harness-level refusal, not a value error at a call site: the
        # question it answers is "can this batch be collected?", not "was this argument
        # valid?". Translated so the CLI exits 2 with a message instead of a traceback.
        raise SeedPlanError(str(e)) from e
    _assert_seed_policy_usable(cfg, n_rollouts=per_task)
    seeds = {(str(e["task_id"]), int(e["rollout_index"])): int(e["sampling_seed"]) for e in plan}  # type: ignore[arg-type]
    log.info(
        "seed plan: base_seed=%d policy=%s derivation=%s %d rollout(s) digests=%s",
        cfg.seed,
        cfg.seed_policy,
        cfg.seed_derivation,
        len(plan),
        plan_digest(plan),
    )

    # Resume: rollouts already persisted by a previous attempt of this same run are copied
    # forward rather than re-sampled. The persisted seed is re-derived and compared, because
    # a row carrying a seed this run would not have produced means the batch is a mixture of
    # two derivations - reproducible as neither.
    persisted = _load_persisted(cfg) if resume else {}
    if persisted:
        log.info("resuming: %d rollout(s) already persisted", len(persisted))

    served = check_server(cfg)
    log.info("server %s serves %s", cfg.base_url, served)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    reused = 0

    started = time.monotonic()
    for task_id in task_ids:
        task = find_task(task_id, split=cfg.split)
        for index in range(per_task):
            rollout_id = f"{cfg.run_id}__{task_id}__r{index:02d}"

            if rollout_id in persisted:
                row = persisted[rollout_id]
                expected = seeds[(task_id, index)]
                recorded = (row.get("agent") or {}).get("sampling_seed")
                if recorded is not None and int(recorded) != expected:
                    raise CollectorError(
                        f"resume mismatch for {rollout_id}: persisted sampling_seed {recorded} but this run "
                        f"derives {expected} from base_seed={cfg.seed} under {cfg.seed_derivation}. "
                        "Re-deriving would mix two derivations in one batch; start a new run_id instead."
                    )
                rows.append(row)
                reused += 1
                log.info("[resume] %s kept from a previous attempt", rollout_id)
                continue

            log.info(
                "[%s/%s] %s rollout %d/%d", task_ids.index(task_id) + 1, len(task_ids), task_id, index + 1, per_task
            )
            deadline = time.monotonic() + cfg.timeout_s_per_rollout
            outcome = run_one_rollout(
                cfg=cfg,
                client=_client(cfg.base_url, cfg.api_key),
                task=task,
                rollout_id=rollout_id,
                rollout_index=index,
                deadline=deadline,
                tokenizer=tokenizer,
            )
            rows.append(outcome.row)
            private.append(
                {
                    "rollout_id": rollout_id,
                    "task_id": task_id,
                    "tool_calls": outcome.tool_calls,
                    "stopped_on": outcome.stopped_on,
                    "reward_eligible": outcome.reward_eligible,
                    "context_budget": outcome.context_budget,
                    "compaction_events": outcome.compaction_events,
                    "infrastructure_failure": outcome.infrastructure_failure,
                }
            )
            if outcome.stopped_on in {"error", "infrastructure_error"}:
                failures.append(
                    {
                        "rollout_id": rollout_id,
                        "task_id": task_id,
                        "stopped_on": outcome.stopped_on,
                        "error": outcome.row.get("error"),
                        "infrastructure_failure": outcome.infrastructure_failure,
                    }
                )

    # Reward statistics are computed over reward-eligible rows only. An infrastructure
    # failure has no reward, so counting it would fabricate an episode.
    eligible = reward_eligible_rows(rows)
    rewards = [row["result"]["reward"] for row in eligible]
    distinct = sorted(set(rewards))
    infra_rows = [row for row in rows if is_infrastructure_failure(row)]
    per_task_valid = {task_id: sum(1 for row in eligible if row["task"]["id"] == task_id) for task_id in task_ids}
    summary = {
        "run_id": cfg.run_id,
        "schema_version": SCHEMA_VERSION,
        "model": cfg.model_name,
        "model_revision": cfg.model_revision,
        "checkpoint_id": None,
        "prompt_config_version": cfg.prompt_version,
        "tools_version": cfg.tools_version,
        "split": cfg.split,
        "tasks": task_ids,
        "rollouts_per_task": per_task,
        # ---- seed provenance (Phase 6H) --------------------------------------------
        # The summary is where a reader looks to decide whether a batch is reproducible,
        # so the derivation is stated here rather than left to be inferred from the rows.
        # `per_rollout` is written explicitly: a summary claiming 4 rollouts while all four
        # shared one seed (Phase 6G) was technically consistent and completely misleading.
        "sampling_seeds": {
            "base_seed": cfg.seed,
            "sampling_seed_derivation": cfg.seed_derivation,
            "seed_policy": cfg.seed_policy,
            "per_rollout": cfg.seed_policy != "literal",
            "n_distinct_sampling_seeds": len({int(row["agent"]["sampling_seed"]) for row in rows if row.get("agent")}),
            "plan_digest": plan_digest(plan),
            # The full mapping, not just the digest: a digest cannot be inverted, and the
            # first question about an unreproducible batch is "which seed did rollout 2 of
            # this task use?". Small enough to inline for a batch of this size.
            "plan": plan,
            "resumed_rollouts": reused,
        },
        "n_rollouts": len(rows),
        "n_reward_eligible": len(eligible),
        "n_infrastructure_invalid": len(infra_rows),
        "n_context_overflow": sum(
            1
            for row in infra_rows
            if (row.get("infrastructure_failure") or {}).get("error_type")
            in {"context_budget_exceeded", "tokenizer_unavailable"}
        ),
        "per_task_valid_rollouts": per_task_valid,
        "tasks_with_zero_valid_rollouts": sorted(t for t, n in per_task_valid.items() if n == 0),
        "n_positive_reward": sum(1 for r in rewards if r is not None and r > 0),
        "diagnostic_only": cfg.diagnostic_only,
        "baseline_valid": False,
        "wall_clock_s": round(time.monotonic() - started, 3),
        "distinct_rewards": distinct,
        "all_termination_reasons": sorted({r["termination_reason"] for r in rows}),
        "context_budget": _first_context_budget(rows),
        # Lifecycle stages, filled in after persistence below. Declared here so every
        # summary carries the same keys whether or not anything went wrong.
        "collection_completed": False,
        "evidence_persisted": None,
        "baseline_acceptance": None,
        "training_suitability": None,
    }
    summary["baseline_valid"], summary["validity_problems"] = _assess_validity(
        cfg=cfg,
        summary=summary,
        eligible=eligible,
        rewards=rewards,
    )

    # ---- persist, then judge -------------------------------------------------------
    # Phase 6E proved why the order matters. The uniform-reward guard raised before
    # anything was written, so a completed batch of four rollouts - roughly 80 seconds of
    # GPU time - left no evidence at all, and the only surviving record was the guard's own
    # error message. The batch that most needs inspecting is exactly the one a guard rejects,
    # so the guard must never be the thing that destroys it.
    #
    # The four stages are now distinct and each is recorded in the summary:
    #   collection_completed  - the loop finished over every configured rollout
    #   evidence_persisted    - every public/private row reached disk and was fsynced
    #   baseline_acceptance   - the run-level guards' verdict
    #   training_suitability  - always false here; stated, not inferred
    persisted = _persist_evidence(cfg, rows, private, summary)
    summary["collection_completed"] = True
    summary["evidence_persisted"] = persisted
    summary["baseline_acceptance"] = {
        "passed": summary["baseline_valid"],
        "problems": list(summary["validity_problems"]),
    }
    # A diagnostic-only run is not training evidence by construction, and neither is a run
    # whose guards failed. Both are stated rather than left for a reader to derive.
    summary["training_suitability"] = {
        "training_ready": False,
        "optimizer_allowed": False,
        "reason": (
            "diagnostic_only run: collected for diagnosis, never for training"
            if cfg.diagnostic_only
            else "run-level acceptance guards failed"
            if not summary["baseline_valid"]
            else "collection alone does not establish training suitability"
        ),
    }
    summary["rollouts_path"] = str(cfg.output_path("rollouts"))
    summary_path = cfg.output_path("rollouts").parent / "collection_summary.json"
    _write_json_atomic(summary_path, summary)
    log.info("wrote %d rows to %s", len(rows), cfg.output_path("rollouts"))

    # Only now, with the evidence durable, is it legitimate to refuse the batch.
    if cfg.require_reward_variance and not allow_uniform:
        if len(distinct) < cfg.min_distinct_rewards:
            raise UniformRewardError(
                f"all {len(eligible)} reward-eligible rollouts scored from {distinct} "
                f"distinct reward value(s); 6.2 requires >= {cfg.min_distinct_rewards}. A "
                "constant reward gives GRPO no gradient, so this is a harness result, not a "
                f"baseline. The {len(rows)} completed rollout(s) are preserved at "
                f"{summary_path.parent} - inspect them before re-running.\n"
                "Investigate before proceeding. `--allow-uniform` overrides for a smoke."
            )

    # Run-level guards are evaluated after the artifacts exist, so an invalid batch is
    # still on disk to inspect - but it is loudly labelled and the caller gets a
    # non-zero exit rather than a summary that looks like a baseline.
    if not summary["baseline_valid"]:
        message = (
            "collection produced evidence that is NOT a valid baseline:\n  - "
            + "\n  - ".join(summary["validity_problems"])
            + f"\nArtifacts were written to {summary_path.parent} for inspection."
        )
        if cfg.diagnostic_only:
            log.warning("%s\n(diagnostic_only is set: run is labelled non-training evidence)", message)
        else:
            raise InfrastructureNotReadyError(message, report=summary)
    return summary


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON via a temporary file and `os.replace`, so a reader never sees a half file.

    `collection_summary.json` is rewritten as the run progresses and is the file a human
    opens first when a batch is rejected. A truncated summary would be read as an empty
    run, which is a worse failure than the one it was meant to report.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return path


def _persist_evidence(
    cfg: BaselineConfig,
    rows: list[dict[str, Any]],
    private: list[dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Write every completed row to disk and flush it before any guard is consulted.

    Returns a small report of what reached disk, which the summary carries so "collection
    completed" and "evidence persisted" stay separately checkable.

    The public rows go through `write_jsonl`, which validates each row against the schema
    as it is appended: a malformed row is a bug found now rather than during analysis. The
    private log is written by `secure_write_private_log`, which creates it 0600 - the
    Phase 6E artifact landed at 0664 because it came from a plain `open()`.
    """
    rollouts_path = cfg.output_path("rollouts")
    private_path = cfg.output_path("private")

    write_jsonl(rows, rollouts_path)
    write_jsonl(
        [r for r in rows if r["termination_reason"] in {"error", "infrastructure_error"}],
        cfg.output_path("failures"),
    )
    # The private log holds tool arguments and raw outputs, which are not trace-row shaped
    # (and must never be validated as such).
    secure_write_private_log(private_path, private)

    for path in (rollouts_path, private_path):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())

    public_lines = sum(1 for _ in rollouts_path.open(encoding="utf-8"))
    private_lines = sum(1 for _ in private_path.open(encoding="utf-8"))
    mode = stat.S_IMODE(private_path.stat().st_mode)
    report = {
        "persisted": True,
        "rollouts_path": str(rollouts_path),
        "n_public_rows": public_lines,
        "private_path": str(private_path),
        "n_private_rows": private_lines,
        "private_mode": f"{mode:04o}",
        "run_dir": str(rollouts_path.parent),
    }
    if public_lines != len(rows):
        raise CollectorError(
            f"{rollouts_path} holds {public_lines} row(s) but {len(rows)} were collected; "
            "refusing to report a batch whose evidence is incomplete"
        )
    if private_lines != len(private):
        raise CollectorError(
            f"{private_path} holds {private_lines} row(s) but {len(private)} were collected; "
            "refusing to report a batch whose private evidence is incomplete"
        )
    return report


def _public_context_telemetry(probe: Probe | None, compaction_events: int) -> dict[str, Any] | None:
    """The public, privacy-safe context-budget record for one rollout.

    Counts only, and the key set is exactly what `analysis/trace_schema.json` declares for
    `context_budget` (which is `additionalProperties: false`, so this is not advisory).

    Two shape notes, both deliberate:

    * The schema's vocabulary is `capacity` / `peak_prompt_tokens`; `Probe.to_dict()` uses
      `max_model_len`-derived arithmetic and calls the same quantity `prompt_tokens`. The
      schema is the contract for a persisted row, so the schema's names win and the probe's
      values are mapped onto them. Emitting `Probe.to_dict()` verbatim would fail
      validation on an undeclared key, which is how this defect stayed hidden: nothing
      wrote the field, so nothing was ever rejected.
    * `status` is added to the schema rather than dropped, because "did this rollout come
      near the wall" is the whole point of the record, and `ok` versus `context_budget_exceeded`
      is that answer.

    Returns None when no probe ran (the rollout died before its first measurement). Null is
    the honest value: a zeroed record would claim a measurement of zero.
    """
    if probe is None:
        return None
    return {
        "max_model_len": probe.budget.max_model_len,
        "reserved_completion_tokens": probe.budget.reserved_completion_tokens,
        "safety_margin_tokens": probe.budget.safety_margin_tokens,
        "capacity": probe.budget.capacity,
        "peak_prompt_tokens": probe.prompt_tokens if probe.prompt_tokens is not None else 0,
        "compaction_events": compaction_events,
        "status": probe.status,
        "context_remaining_tokens": probe.remaining,
    }


def _first_context_budget(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The first row's context-budget telemetry, for the run summary.

    Counts only, so this is safe to print and to commit alongside the summary.

    A row that predates the public telemetry field simply has none; it is skipped rather
    than treated as an error, so historical batches stay summarisable.
    """
    for row in rows:
        budget = row.get("context_budget")
        if isinstance(budget, dict):
            return budget
    return None


def _assess_validity(
    *,
    cfg: BaselineConfig,
    summary: dict[str, Any],
    eligible: list[dict[str, Any]],
    rewards: list[Any],
) -> tuple[bool, list[str]]:
    """Apply the run-level guards and return `(baseline_valid, problems)`.

    Each guard answers a question a bare rollout count cannot: did every configured task
    contribute a reward-eligible episode, did context overflow wipe a task out, and is
    there any signal for GRPO to learn from. `diagnostic_only` permits the last one to be
    a warning - it never makes an invalid run trainable.
    """
    problems: list[str] = []

    zero_tasks = summary.get("tasks_with_zero_valid_rollouts") or []
    if zero_tasks:
        problems.append(
            f"tasks with zero reward-eligible rollouts: {', '.join(zero_tasks)} "
            "(every attempt failed before or at preflight)"
        )

    if summary["n_context_overflow"] and not cfg.diagnostic_only:
        problems.append(
            f"{summary['n_context_overflow']} rollout(s) exceeded the context budget; "
            "those attempts never executed the model"
        )

    if summary["n_reward_eligible"] < cfg.min_valid_rollouts:
        problems.append(
            f"only {summary['n_reward_eligible']} reward-eligible rollout(s), "
            f"config requires at least {cfg.min_valid_rollouts}"
        )

    if len(set(rewards)) < cfg.min_distinct_rewards:
        problems.append(
            f"reward-eligible rollouts produced {len(set(rewards))} distinct reward "
            f"value(s) {sorted(set(rewards))}; at least {cfg.min_distinct_rewards} are "
            "required for a GRPO signal"
        )

    # A run can never be both diagnostic-only and training-valid.
    #
    # This is the structural guarantee behind `diagnostic_only`: it is not a label applied
    # to the output, it is a veto on the verdict. A diagnostic run cannot pass acceptance
    # even if the batch happens to satisfy every numeric guard, so no later code path can
    # promote one to a baseline by accident.
    if cfg.diagnostic_only:
        problems.append("diagnostic_only run: collected for diagnosis, never training evidence")

    return (not problems), problems


# ------------------------------------------------------------------------- CLI


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 6.2 - collect baseline rollouts from the frozen control arm.",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--tasks", type=int, default=None, help="use only the first N tasks")
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help=(
            "comma-separated task ids to collect, overriding --tasks. Needed because the "
            "interesting task is not always the first one: the Phase 6 overflow was on "
            "testupdate_money_api_v2, which is last in task_subset."
        ),
    )
    parser.add_argument("--rollouts", type=int, default=None, help="override rollouts per task")
    parser.add_argument(
        "--allow-uniform",
        action="store_true",
        help="do not fail when every rollout scores the same (smoke runs only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="check the server and the task list, collect nothing")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "keep rollouts this run_id already persisted instead of re-sampling them; each "
            "kept row's recorded seed is re-derived and must match, so a resumed batch is "
            "still one derivation rather than two stitched together"
        ),
    )
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "collect for diagnosis even if the run-level guards fail; the summary is "
            "labelled diagnostic_only and can never be training evidence"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="override the tokenizer path used for context-budget preflight",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cfg = BaselineConfig.load(args.config)

    # CLI overrides exist for diagnosis on unusual hardware, not to change the protocol.
    #
    # These used to be assignments (`cfg.diagnostic_only = True`), which crashed with
    # FrozenInstanceError: the config is frozen on purpose, so that the record of what a run
    # used cannot be edited after it started. `dataclasses.replace` keeps that property - the
    # frozen instance is never mutated, a new one is built - and keeps the type honest, so
    # the dataclass does not need to be made mutable to support a command-line flag.
    overrides: dict[str, Any] = {}
    if args.diagnostic_only:
        overrides["diagnostic_only"] = True
    if args.tokenizer is not None:
        overrides["tokenizer_path"] = args.tokenizer
    if overrides:
        cfg = _with_overrides(cfg, **overrides)
    if cfg.diagnostic_only:
        log.warning("diagnostic_only mode: this run's artifacts must never be used as a baseline or as training data")

    task_ids = None
    if args.task_ids:
        task_ids = [part.strip() for part in args.task_ids.split(",") if part.strip()]
        unknown = [t for t in task_ids if t not in cfg.task_subset]
        if unknown:
            print(
                f"ERROR: task id(s) not in this run's task_subset: {', '.join(unknown)}",
                file=sys.stderr,
            )
            return 2

    # A SIGTERM from a `timeout` guard should stop the batch between rollouts, not in the
    # middle of one, so that the rollouts already written stay valid.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    if args.dry_run:
        served = check_server(cfg)
        ids = task_ids if task_ids else cfg.task_subset
        for task_id in ids:
            find_task(task_id, split=cfg.split)
        # Report the context budget the run would enforce, so a misconfigured tokenizer
        # path is visible before collection starts rather than on the first rollout.
        tokenizer_ready = True
        tokenizer_error = None
        try:
            load_tokenizer(cfg.tokenizer_path, revision=cfg.model_revision)
        except TokenizerUnavailableError as exc:
            tokenizer_ready = False
            tokenizer_error = str(exc)
        budget = cfg.context_budget()
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "server": cfg.base_url,
                    "served_model": served,
                    "revision": cfg.model_revision,
                    "tasks": ids,
                    "n_tasks": len(ids),
                    "rollouts_per_task": args.rollouts or cfg.rollouts_per_task,
                    "tokenizer_ready": tokenizer_ready,
                    "tokenizer_error": tokenizer_error,
                    "tokenizer_path": str(cfg.tokenizer_path),
                    "context_budget": budget.to_dict(),
                    "prompt_capacity_tokens": budget.capacity,
                    "diagnostic_only": cfg.diagnostic_only,
                },
                indent=2,
            )
        )
        return 0

    try:
        summary = collect(
            cfg,
            tasks_limit=args.tasks,
            rollouts_override=args.rollouts,
            allow_uniform=args.allow_uniform,
            task_ids_override=task_ids,
            resume=args.resume,
        )
    except CollectorError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if isinstance(e, InfrastructureNotReadyError) and e.report:
            print(json.dumps(e.report, indent=2), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(
            f"{summary['n_rollouts']} rollouts over {len(summary['tasks'])} tasks; "
            f"{len(summary['distinct_rewards'])} distinct reward value(s) "
            f"{summary['distinct_rewards']} in {summary['wall_clock_s']}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
