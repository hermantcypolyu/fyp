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

    `observation_char_cap` bounds any single retained message and is the *starting* bound:
    when a `budget` and `tokenizer` are supplied, compaction halves it (never below
    `min_char_cap`) until the prompt measures within budget. `keep_recent_tool_messages` is
    the number of newest tool results kept verbatim; older ones are replaced by a
    fixed-shape metadata line, and the retained count shrinks under pressure.
    """

    observation_char_cap: int = 2000
    keep_recent_tool_messages: int = 4
    #: The smallest per-message character cap compaction will shrink to. Below this a
    #: retained observation stops being useful, so compaction drops it entirely instead.
    min_char_cap: int = 200


@dataclass
class CompactionResult:
    messages: list[dict[str, Any]]
    compacted: bool
    replaced_observations: int
    chars_before: int
    chars_after: int
    reasons: list[str] = field(default_factory=list)
    #: How many elision/clip passes ran. More than one means a single pass did not fit.
    passes: int = 1
    #: Whether the returned messages measure within `budget`. `None` when no tokenizer or
    #: budget was supplied, so no claim is made either way.
    within_budget: bool | None = None
    #: The measured prompt tokens of the returned messages, when measurement was possible.
    prompt_tokens: int | None = None


def _content_chars(message: dict[str, Any]) -> int:
    """Characters in a message's own content, ignoring tool-call arguments."""
    return len(str(message.get("content") or ""))


def compact_messages(
    messages: Sequence[dict[str, Any]],
    *,
    policy: CompactionPolicy | None = None,
    tokenizer: Any = None,
    budget: ContextBudget | None = None,
) -> CompactionResult:
    """Shrink a conversation deterministically until it measures within `budget`.

    **This function enforces a hard token limit.** When `tokenizer` and `budget` are both
    supplied, the result is re-measured with :func:`measure` and the elision/clipping
    passes repeat until the prompt fits or no further reduction is possible. The
    `within_budget` field on the result reports which happened, so a caller can turn "did
    not fit" into an infrastructure decision instead of discovering it at the server.

    That loop is the fix for a real defect. This function used to take `budget` and
    `tokenizer` and use neither: it clipped each message to a fixed character cap, ran
    exactly one pass, and returned without ever measuring the outcome. A single pass can
    leave a prompt far above the limit, because a character cap is not a token budget -
    2000 characters of dense source can exceed 700 tokens - and because a model that
    narrates without calling tools produces many messages that are each under the cap and
    therefore each retained. The rollout then failed with
    `stage: preflight_after_compaction` alleging the request "needs N tokens", even though
    nothing in the compaction had attempted to enforce N.

    Preserved unconditionally:
      * the system message (harness rules and the JSON tool contract)
      * the task instruction (the first user turn)
      * every assistant message that emitted `tool_calls`, together with its tool replies,
        so ids stay paired

    Reduced, in increasing order of severity across passes:
      * older `tool` observations are elided to a one-line metadata stub, then the newest
        ones are clipped, then the character cap itself is halved
      * assistant prose turns without tool calls are clipped, then elided entirely

    Everything here is deterministic: same input and same tokenizer give the same output.
    Nothing is summarised by a model and no tool call is removed from the transcript.
    """
    policy = policy or CompactionPolicy()
    # The hard limit is only enforceable when both halves of the measurement exist. Without
    # them this degrades to the old character-only behaviour, and says so, rather than
    # silently claiming to have enforced a budget it never looked at.
    enforce = tokenizer is not None and budget is not None

    original = [dict(m) for m in messages]
    system_msgs = [m for m in original if m.get("role") == "system"]
    others = [m for m in original if m.get("role") != "system"]

    first_user_index = next((i for i, m in enumerate(others) if m.get("role") == "user"), None)
    instruction = others[first_user_index] if first_user_index is not None else None
    tail = [m for i, m in enumerate(others) if i != first_user_index]

    before = sum(_content_chars(m) for m in original)
    chars_before_all = before

    reasons: list[str] = []
    replaced = 0
    cap = policy.observation_char_cap
    keep_recent = policy.keep_recent_tool_messages
    elide_prose = False
    passes = 0
    #: Upper bound on passes. Each pass strictly lowers `cap` or `keep_recent` or turns on
    #: prose elision, so this cannot spin; the bound exists so a pathological input still
    #: terminates with `within_budget: False` instead of looping.
    max_passes = 12

    result = list(system_msgs)
    if instruction is not None:
        result.append(instruction)

    while True:
        tool_positions = [i for i, m in enumerate(tail) if m.get("role") == "tool"]
        keep_from = tool_positions[-keep_recent:] if keep_recent > 0 and tool_positions else []
        keep_set = set(keep_from)

        rebuilt_tail: list[dict[str, Any]] = []
        replaced = 0
        for index, message in enumerate(tail):
            role = message.get("role")
            clipped = dict(message)
            content = message.get("content")

            if role == "tool":
                text = "" if content is None else str(content)
                if index not in keep_set:
                    clipped["content"] = f"[tool output elided by context compaction: {len(text)} chars]"
                    replaced += 1
                elif len(text) > cap:
                    kept = _clip(text, cap)
                    clipped["content"] = kept + f"\n[... {len(text) - len(kept)} chars truncated ...]"
                    replaced += 1
            elif role == "assistant":
                text = "" if content is None else str(content)
                if message.get("tool_calls"):
                    # Prose attached to a tool call is compressible even though the call
                    # itself is not. The call must survive (its id is paired with the tool
                    # reply, and dropping either side is a server-side 400), but the words
                    # before it are the model's commentary and are bounded the same way as
                    # a prose-only turn. Leaving this unclipped was a residual leak: a
                    # model narrating 800 characters before every call kept ~10000 tokens
                    # of prose that compaction could see and did not touch.
                    if elide_prose:
                        if text:
                            clipped["content"] = f"[assistant message elided by context compaction: {len(text)} chars]"
                            replaced += 1
                    elif len(text) > cap:
                        kept = _clip(text, cap)
                        clipped["content"] = kept + f"\n[... {len(text) - len(kept)} chars truncated ...]"
                        replaced += 1
                else:
                    text = "" if content is None else str(content)
                    if elide_prose:
                        if text:
                            clipped["content"] = f"[assistant message elided by context compaction: {len(text)} chars]"
                            replaced += 1
                    elif len(text) > cap:
                        kept = _clip(text, cap)
                        clipped["content"] = kept + f"\n[... {len(text) - len(kept)} chars truncated ...]"
                        replaced += 1
            rebuilt_tail.append(clipped)

        result = list(system_msgs)
        if instruction is not None:
            result.append(instruction)
        result.extend(rebuilt_tail)
        passes += 1

        if not enforce:
            break
        if preflight(result, tokenizer, budget).admissible:
            break
        if passes >= max_passes:
            break

        # Escalate. Each branch tightens the knobs that a single pass left loose, in the
        # order that loses the least information first: drop the oldest retained
        # observation, then shrink the per-message cap, then drop prose entirely.
        if keep_recent > 0:
            keep_recent -= 1
        elif cap > policy.min_char_cap:
            cap = max(policy.min_char_cap, cap // 2)
        elif not elide_prose:
            elide_prose = True
        else:
            break

    if replaced:
        reasons.append(f"elided_or_clipped_{replaced}_message(s)")
    if passes > 1:
        reasons.append(f"compaction_passes_{passes}")

    after = sum(_content_chars(m) for m in result)
    within_budget: bool | None = None
    prompt_tokens: int | None = None
    if enforce:
        probe = preflight(result, tokenizer, budget)
        prompt_tokens = probe.prompt_tokens
        within_budget = probe.admissible
        if not within_budget:
            reasons.append("could_not_fit_budget")

    return CompactionResult(
        messages=result,
        compacted=bool(replaced) and after < chars_before_all,
        replaced_observations=replaced,
        chars_before=chars_before_all,
        chars_after=after,
        reasons=reasons,
        passes=passes,
        within_budget=within_budget,
        prompt_tokens=prompt_tokens,
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
