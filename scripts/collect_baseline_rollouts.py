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
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
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
from analysis.trace import SCHEMA_VERSION, Timer, build_row, collect_patch, write_jsonl  # noqa: E402
from envs.coding_env.session import EpisodeSession, find_task  # noqa: E402
from envs.coding_env.tools import TOOL_DEFINITIONS  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "eval" / "baseline_phase6.yaml"

log = logging.getLogger("phase6.collect")


# --------------------------------------------------------------------------- errors


class CollectorError(RuntimeError):
    """A precondition the collector refuses to run without."""


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
    #: Context-budget telemetry for the first request of this rollout.
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
    """
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
                    seed=cfg.seed,
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
        random_seed=cfg.seed,
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

    return RolloutOutcome(
        row=row,
        tool_calls=private_tool_log,
        stopped_on=row["termination_reason"],
        transcript_chars=len(json.dumps(messages)),
        infrastructure_failure=infra,
        context_budget=first_probe.to_dict() if first_probe is not None else None,
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
) -> dict[str, Any]:
    """Collect the batch and return a summary. Raises `UniformRewardError` if required.

    Raises `InfrastructureNotReadyError` before the first request when no tokenizer is
    available, and after the loop when a run-level guard shows the batch is not evidence.
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

    served = check_server(cfg)
    log.info("server %s serves %s", cfg.base_url, served)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []

    started = time.monotonic()
    for task_id in task_ids:
        task = find_task(task_id, split=cfg.split)
        for index in range(per_task):
            rollout_id = f"{cfg.run_id}__{task_id}__r{index:02d}"
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
        "context_budget": None if not rows else _first_context_budget(rows),
    }
    summary["baseline_valid"], summary["validity_problems"] = _assess_validity(
        cfg=cfg,
        summary=summary,
        eligible=eligible,
        rewards=rewards,
    )
    if cfg.require_reward_variance and not allow_uniform:
        if len(distinct) < cfg.min_distinct_rewards:
            raise UniformRewardError(
                f"all {len(eligible)} reward-eligible rollouts scored from {distinct} "
                f"distinct reward value(s); 6.2 requires >= {cfg.min_distinct_rewards}. A "
                "constant reward gives GRPO no gradient, so this is a harness result, not a "
                "baseline:\n"
                f"  {json.dumps(summary, indent=2)}\n"
                "Investigate before proceeding. `--allow-uniform` overrides for a smoke."
            )

    write_jsonl(rows, cfg.output_path("rollouts"))
    write_jsonl(
        [r for r in rows if r["termination_reason"] in {"error", "infrastructure_error"}],
        cfg.output_path("failures"),
    )
    # The private log holds tool arguments and raw outputs, which are not trace-row shaped
    # (and must never be validated as such). It is written directly, not via write_jsonl.
    private_path = cfg.output_path("private")
    private_path.parent.mkdir(parents=True, exist_ok=True)
    with private_path.open("w", encoding="utf-8") as handle:
        for entry in private:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    summary["rollouts_path"] = str(cfg.output_path("rollouts"))
    summary_path = cfg.output_path("rollouts").parent / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %d rows to %s", len(rows), cfg.output_path("rollouts"))

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


def _first_context_budget(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The first row's context-budget telemetry, for the run summary.

    Counts only, so this is safe to print and to commit alongside the summary.
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
    if args.diagnostic_only:
        cfg.diagnostic_only = True
    if args.tokenizer is not None:
        cfg.tokenizer_path = args.tokenizer
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
