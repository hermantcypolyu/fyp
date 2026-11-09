"""Deterministic context budgeting for rollout collection.

Phase 6 bring-up lost eight rollouts to `HTTP 400: maximum context length is 8192
tokens`. Those rows were recorded as ordinary model failures, which is wrong on two
counts: the model never produced a reward-eligible episode, and every one of the eight
failures came from the same task, so one sixth of the batch silently contributed
nothing to the reward distribution.

This module makes context arithmetic explicit and deterministic:

* :func:`measure` turns a chat-message list into an exact token count using the pinned
  model tokenizer. There is no character-count fallback, because an approximation is
  indistinguishable from a measurement once it is written into a trace row.
* :func:`preflight` decides whether a request is admissible and reports the numbers it
  used, so the decision can be re-derived from the row alone.
* :func:`compact_messages` shrinks a conversation that has grown past its budget,
  preserving the system rules, the task instruction and the model's own submitted patch
  state, and replacing only older bulky tool observations with deterministic metadata.

Everything here is pure and CPU-only: no network, no GPU, no model server.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("analysis.context_budget")

__all__ = [
    "BudgetError",
    "TokenizerUnavailableError",
    "ContextBudget",
    "Probe",
    "DEFAULT_SAFETY_MARGIN_TOKENS",
    "DEFAULT_RESERVED_COMPLETION_TOKENS",
    "INFRA_ERROR_CONTEXT_BUDGET",
    "INFRA_ERROR_TOKENIZER_MISSING",
    "STATUS_OK",
    "STATUS_EXCEEDED",
    "STATUS_TOKENIZER_UNAVAILABLE",
    "load_tokenizer",
    "measure",
    "preflight",
    "compact_messages",
    "infrastructure_failure",
]

#: Two counts that are always present in a verdict.
STATUS_OK = "ok"
STATUS_EXCEEDED = "context_budget_exceeded"
STATUS_TOKENIZER_UNAVAILABLE = "tokenizer_unavailable"

#: Error type written into infrastructure-failure records for a budget breach.
INFRA_ERROR_CONTEXT_BUDGET = "context_budget_exceeded"
INFRA_ERROR_TOKENIZER_MISSING = "tokenizer_unavailable"

#: Head-room for the chat template's per-message framing, which the tokenizer cannot see
#: when it is handed a flattened string. Qwen2.5 adds a handful of control tokens per
#: message; 64 covers the longest prompts this harness builds with a wide margin.
DEFAULT_SAFETY_MARGIN_TOKENS = 64

#: Completion space reserved for the model's next reply. Mirrors the sampling config.
DEFAULT_RESERVED_COMPLETION_TOKENS = 1024


class BudgetError(Exception):
    """A context-budget decision could not be made, or the request was inadmissible."""


class TokenizerUnavailableError(BudgetError):
    """The pinned tokenizer could not be loaded.

    This is raised rather than approximated. Collection must not proceed on a guess:
    a character-count estimate that happens to be 20% low would let an over-long
    request through and reproduce exactly the Phase 6 bring-up failure, but with a
    measurement-shaped number recorded as if it were exact.
    """


# ---------------------------------------------------------------- measurement


def load_tokenizer(model_path: str | Path | None = None, *, revision: str | None = None) -> Any:
    """Load the pinned tokenizer, or raise :class:`TokenizerUnavailableError`.

    `transformers` is optional and lives in the inference virtualenv, not the project
    one. A caller that cannot import it gets a clear, actionable failure instead of a
    silent downgrade to characters.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 - any import failure is "not ready"
        raise TokenizerUnavailableError(
            "transformers is not importable, so prompt tokens cannot be measured exactly. "
            "Install it in this environment or point PYTHONPATH at the inference venv. "
            f"Refusing to approximate with character counts. ({exc})"
        ) from exc

    try:
        if model_path is None:
            raise TokenizerUnavailableError("no tokenizer path supplied")
        # `expanduser` because the config writes the cache location the way a shell does.
        # Without it a `~/...` path is a relative directory literally named "~", which
        # does not exist, and the failure would look like a missing model rather than a
        # missing expansion.
        path = Path(model_path).expanduser()
        if not path.exists():
            raise TokenizerUnavailableError(f"tokenizer path does not exist: {path}")
        return AutoTokenizer.from_pretrained(str(path), revision=revision, local_files_only=True)
    except TokenizerUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 - unreadable tokenizer is "not ready"
        raise TokenizerUnavailableError(f"tokenizer could not be loaded from {model_path!r}: {exc}") from exc


def _message_texts(messages: Sequence[dict[str, Any]]) -> list[str]:
    """Flatten an OpenAI-style message list into text the tokenizer can count.

    Tool calls are rendered back to JSON because that is what occupies the model's
    context: the arguments it emitted, not just the prose it wrote.
    """
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        text = "" if content is None else str(content)
        parts.append(f"<|{role}|>\n{text}")
        for call in message.get("tool_calls") or []:
            fn = (call or {}).get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args, sort_keys=True)
            parts.append(f"<|tool_call|>{name}{args}")
    return parts


def measure(messages: Sequence[dict[str, Any]], tokenizer: Any) -> int:
    """Exact prompt-token count for a message list.

    Raises :class:`TokenizerUnavailableError` when `tokenizer` is None: an unmeasurable
    prompt is an infrastructure problem, never a zero-token prompt.
    """
    if tokenizer is None:
        raise TokenizerUnavailableError(
            "measure() requires a tokenizer; refusing to estimate prompt tokens from characters"
        )
    payload = "\n".join(_message_texts(messages))
    encoded = tokenizer(payload, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return len(ids)


# ---------------------------------------------------------------- the budget


@dataclass(frozen=True)
class ContextBudget:
    """The four numbers that decide admissibility.

    `max_model_len` is the server's `--max-model-len`. `reserved_completion_tokens` is
    what the sampling config will ask for. `safety_margin_tokens` covers chat-template
    framing. A request is admissible when the three sum to no more than the limit.
    """

    max_model_len: int = 8192
    reserved_completion_tokens: int = DEFAULT_RESERVED_COMPLETION_TOKENS
    safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS

    def __post_init__(self) -> None:
        if self.max_model_len <= 0:
            raise BudgetError(f"max_model_len must be positive, got {self.max_model_len}")
        if self.reserved_completion_tokens < 0:
            raise BudgetError(f"reserved_completion_tokens must be >= 0, got {self.reserved_completion_tokens}")
        if self.safety_margin_tokens < 0:
            raise BudgetError(f"safety_margin_tokens must be >= 0, got {self.safety_margin_tokens}")
        total = self.reserved_completion_tokens + self.safety_margin_tokens
        if total >= self.max_model_len:
            raise BudgetError(
                f"reserved completion ({self.reserved_completion_tokens}) plus safety margin "
                f"({self.safety_margin_tokens}) leaves no room in max_model_len "
                f"({self.max_model_len})"
            )

    @property
    def capacity(self) -> int:
        """Prompt tokens available once completion space and margin are reserved."""
        return self.max_model_len - self.reserved_completion_tokens - self.safety_margin_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "max_model_len": self.max_model_len,
            "reserved_completion_tokens": self.reserved_completion_tokens,
            "safety_margin_tokens": self.safety_margin_tokens,
        }


@dataclass(frozen=True)
class Probe:
    """The outcome of measuring one message list against a budget."""

    prompt_tokens: int | None
    budget: ContextBudget
    status: str
    detail: str = ""

    @property
    def admissible(self) -> bool:
        return self.status == STATUS_OK

    @property
    def remaining(self) -> int | None:
        """Prompt tokens still available. `None` when the prompt could not be measured."""
        if self.prompt_tokens is None:
            return None
        return self.budget.capacity - self.prompt_tokens

    def to_dict(self) -> dict[str, Any]:
        """Trace-facing view. Contains counts only - never prompt text."""
        out: dict[str, Any] = {
            "status": self.status,
            "prompt_tokens": self.prompt_tokens,
            "reserved_completion_tokens": self.budget.reserved_completion_tokens,
            "safety_margin_tokens": self.budget.safety_margin_tokens,
            "max_model_len": self.budget.max_model_len,
            "context_remaining_tokens": self.remaining,
        }
        if self.detail:
            out["detail"] = self.detail
        return out


def preflight(
    messages: Sequence[dict[str, Any]],
    tokenizer: Any,
    budget: ContextBudget,
) -> Probe:
    """Decide whether `messages` fits, without contacting the server.

    Returns a probe with `status == "ok"` when
    `prompt_tokens + reserved_completion_tokens + safety_margin_tokens <= max_model_len`.
    A missing tokenizer yields `tokenizer_unavailable` rather than an estimate.
    """
    if tokenizer is None:
        return Probe(
            prompt_tokens=None,
            budget=budget,
            status=STATUS_TOKENIZER_UNAVAILABLE,
            detail="tokenizer unavailable; prompt tokens cannot be measured exactly",
        )
    try:
        prompt_tokens = measure(messages, tokenizer)
    except TokenizerUnavailableError as exc:
        return Probe(
            prompt_tokens=None,
            budget=budget,
            status=STATUS_TOKENIZER_UNAVAILABLE,
            detail=str(exc),
        )

    requested = prompt_tokens + budget.reserved_completion_tokens + budget.safety_margin_tokens
    if requested > budget.max_model_len:
        return Probe(
            prompt_tokens=prompt_tokens,
            budget=budget,
            status=STATUS_EXCEEDED,
            detail=(
                f"request needs {requested} tokens "
                f"({prompt_tokens} prompt + {budget.reserved_completion_tokens} completion "
                f"+ {budget.safety_margin_tokens} margin) but max_model_len is "
                f"{budget.max_model_len}"
            ),
        )
    return Probe(prompt_tokens=prompt_tokens, budget=budget, status=STATUS_OK)


# ---------------------------------------------------------------- compaction


def _clip(text: str, cap_chars: int) -> str:
    """Deterministic prefix clip. Never random, never token-dependent."""
    if len(text) <= cap_chars:
        return text
    return text[:cap_chars]


@dataclass(frozen=True)
class CompactionPolicy:
    """Deterministic knobs for :func:`compact_messages`.

    `keep_recent_tool_messages` is the number of newest tool results kept verbatim. Older
    ones are replaced by a fixed-shape metadata line. `observation_char_cap` bounds any
    single retained observation.
    """

    observation_char_cap: int = 2000
    keep_recent_tool_messages: int = 4


@dataclass
class CompactionResult:
    messages: list[dict[str, Any]]
    compacted: bool
    replaced_observations: int
    chars_before: int
    chars_after: int
    reasons: list[str] = field(default_factory=list)


def compact_messages(
    messages: Sequence[dict[str, Any]],
    *,
    policy: CompactionPolicy | None = None,
    tokenizer: Any = None,
    budget: ContextBudget | None = None,
) -> CompactionResult:
    """Shrink a conversation deterministically, preserving what must survive.

    Preserved unconditionally:
      * the system message (harness rules and the JSON tool contract)
      * the task instruction (the first user turn)
      * every assistant message that emitted `tool_calls`, together with its tool replies,
        so ids stay paired
      * the newest `policy.keep_recent_tool_messages` tool results, verbatim up to
        `policy.observation_char_cap`

    Replaced: older `tool` observations become a one-line metadata stub naming the tool
    and the original size, with the content dropped. This is deterministic truncation,
    not an LLM summary.

    Assistant prose turns without tool calls are the Phase 6 growth source: they are
    clipped to `policy.observation_char_cap` too, because a model that narrates for 500
    tokens a turn with no tool call will otherwise exhaust the window on its own words.
    """
    policy = policy or CompactionPolicy()
    original = [dict(m) for m in messages]
    reasons: list[str] = []

    system_msgs = [m for m in original if m.get("role") == "system"]
    others = [m for m in original if m.get("role") != "system"]

    first_user_index = next((i for i, m in enumerate(others) if m.get("role") == "user"), None)
    instruction = others[first_user_index] if first_user_index is not None else None
    tail = [m for i, m in enumerate(others) if i != first_user_index]

    tool_positions = [i for i, m in enumerate(tail) if m.get("role") == "tool"]
    keep_from = (
        tool_positions[-policy.keep_recent_tool_messages]
        if len(tool_positions) > policy.keep_recent_tool_messages
        else None
    )

    rebuilt_tail: list[dict[str, Any]] = []
    replaced = 0
    for index, message in enumerate(tail):
        role = message.get("role")
        clipped = dict(message)
        content = message.get("content")

        if role == "tool":
            text = "" if content is None else str(content)
            is_recent = keep_from is not None and index >= keep_from
            if not is_recent:
                clipped["content"] = f"[tool output elided by context compaction: {len(text)} chars]"
                replaced += 1
            elif len(text) > policy.observation_char_cap:
                kept = _clip(text, policy.observation_char_cap)
                clipped["content"] = kept + f"\n[... {len(text) - len(kept)} chars truncated ...]"
                replaced += 1
        elif role == "assistant" and not message.get("tool_calls"):
            text = "" if content is None else str(content)
            if len(text) > policy.observation_char_cap:
                kept = _clip(text, policy.observation_char_cap)
                clipped["content"] = kept + f"\n[... {len(text) - len(kept)} chars truncated ...]"
                replaced += 1
        rebuilt_tail.append(clipped)

    result = list(system_msgs)
    if instruction is not None:
        result.append(instruction)
    result.extend(rebuilt_tail)

    before = sum(len(str(m.get("content") or "")) for m in original)
    after = sum(len(str(m.get("content") or "")) for m in result)
    if replaced:
        reasons.append(f"elided_or_clipped_{replaced}_message(s)")

    return CompactionResult(
        messages=result,
        compacted=bool(replaced) and after < before,
        replaced_observations=replaced,
        chars_before=before,
        chars_after=after,
        reasons=reasons,
    )


# ---------------------------------------------------------------- infra failure


def infrastructure_failure(
    *,
    error_type: str,
    stage: str,
    message: str,
    probe: Probe | None = None,
) -> dict[str, Any]:
    """Build the record written in place of a reward row for an infrastructure fault.

    The shape is deliberate: `reward_eligible: false` and `model_executed: false` mean a
    downstream reader cannot mistake this for a scored episode, so it can be excluded
    from reward distributions and GRPO groups without inspecting the message text.
    """
    record: dict[str, Any] = {
        "error_type": error_type,
        "stage": stage,
        "model_executed": False,
        "reward_eligible": False,
        "message": _clip(message, 2000),
    }
    if probe is not None:
        # `Probe.to_dict()` is deliberately NOT used here. Its key set is the trace's
        # top-level `context_budget` vocabulary, while this nested block is a different
        # schema object with `additionalProperties: false`: it requires `capacity` and
        # rejects `reserved_completion_tokens`, `safety_margin_tokens` and
        # `context_remaining_tokens`. Emitting the probe verbatim made every persisted
        # infrastructure-failure row fail validation - a latent defect that stayed hidden
        # only because the prior run wrote no such row, so the path was never exercised.
        record["context_budget"] = {
            "max_model_len": probe.budget.max_model_len,
            "capacity": probe.budget.capacity,
            # A probe that could not measure the prompt has `prompt_tokens: None`; the
            # schema requires an integer, and 0 is the honest floor for "no tokens
            # measured" here because the failure is already recorded in `error_type`.
            "prompt_tokens": probe.prompt_tokens if probe.prompt_tokens is not None else 0,
            "status": probe.status,
        }
        if probe.detail:
            record["context_budget"]["detail"] = _clip(probe.detail, 2000)
    return record


def is_infrastructure_failure(row: dict[str, Any]) -> bool:
    """True when a persisted row is an infrastructure-invalid attempt.

    Recognises all three forms: the collector's explicit `infrastructure_failure` object,
    the `infrastructure_error` termination the Phase 6B collector writes, and a
    server-side context-length rejection that predates this module - so the Phase 6
    bring-up artifact can be reclassified without rewriting any of its rows.
    """
    if row.get("termination_reason") == "infrastructure_error":
        return True
    infra = row.get("infrastructure_failure")
    if isinstance(infra, dict) and infra.get("reward_eligible") is False:
        return True
    error = row.get("error") or {}
    text = str(error.get("message") or "")
    return "maximum context length" in text and "400" in text


def reward_eligible_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows that may enter reward statistics. Excludes infrastructure failures."""
    return [row for row in rows if not is_infrastructure_failure(row)]
