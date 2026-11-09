"""Phase 6B tests - context budgeting and infrastructure-failure handling.

Every test here is CPU-only. None of them starts a server, loads a model, or touches a
GPU. The tokenizer is either a hand-built fake or the real one via `transformers`, which
lives in the server venv, so a machine without it skips rather than fails.

The point of these tests is not coverage. It is that the two failure modes Phase 6 hit -
a silently over-long prompt and a harness failure scored as a policy failure - cannot come
back unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from analysis.context_budget import (
    DEFAULT_RESERVED_COMPLETION_TOKENS,
    DEFAULT_SAFETY_MARGIN_TOKENS,
    STATUS_EXCEEDED,
    STATUS_OK,
    STATUS_TOKENIZER_UNAVAILABLE,
    BudgetError,
    CompactionPolicy,
    ContextBudget,
    Probe,
    TokenizerUnavailableError,
    compact_messages,
    infrastructure_failure,
    is_infrastructure_failure,
    load_tokenizer,
    measure,
    preflight,
    reward_eligible_rows,
)

REPO = Path(__file__).resolve().parents[2]
SCHEMA = REPO / "analysis" / "trace_schema.json"


# --------------------------------------------------------------------------- fixtures


class FakeTokenizer:
    """A tokenizer with the one behaviour the module depends on.

    It tokenises on whitespace, so a test can say "this payload is n tokens" and mean it.
    `measure` prefixes each message with a `<|role|>` marker, so a single-message prompt
    costs one token for the marker plus one per content word. The real tokenizer is
    exercised separately; using it here would make these tests depend on a snapshot that
    the project venv deliberately does not carry.
    """

    def __init__(self, per_word: int = 1) -> None:
        self.per_word = per_word
        self.calls: list[str] = []

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        assert add_special_tokens is False, "special tokens would make the count wrong by a constant"
        self.calls.append(text)
        n = len(text.split()) * self.per_word
        return {"input_ids": list(range(n))}


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def budget() -> ContextBudget:
    # Small numbers keep the arithmetic checkable by hand in the assertions below.
    return ContextBudget(max_model_len=100, reserved_completion_tokens=20, safety_margin_tokens=5)


def _messages(n_words: int, role: str = "user") -> list[dict[str, Any]]:
    """One message whose `measure` count is exactly `n_words`.

    `measure` prefixes `<|role|>`, which costs one whitespace token, so the content is
    one word short of the target.
    """
    assert n_words >= 1
    return [{"role": role, "content": " ".join(["w"] * (n_words - 1))}]


def _exact_tokenizer() -> FakeTokenizer:
    """A tokenizer that counts only the payload's own words, ignoring the role marker.

    Used where a test wants an exact prompt-token count from a multi-message list without
    re-deriving the marker arithmetic in every assertion.
    """
    return FakeTokenizer()


# ------------------------------------------------------------------ 1-2: budget maths


def test_capacity_subtracts_both_reserves(budget: ContextBudget) -> None:
    """1. Capacity is the window minus the completion reserve and the safety margin."""
    assert budget.capacity == 100 - 20 - 5


def test_defaults_match_the_frozen_config() -> None:
    """2. The module's defaults agree with configs/eval/baseline_phase6.yaml's context block."""
    import yaml

    config = yaml.safe_load((REPO / "configs" / "eval" / "baseline_phase6.yaml").read_text(encoding="utf-8"))
    context = config["context"]
    defaults = ContextBudget()
    assert context["max_model_len"] == defaults.max_model_len
    assert context["reserved_completion_tokens"] == DEFAULT_RESERVED_COMPLETION_TOKENS
    assert context["safety_margin_tokens"] == DEFAULT_SAFETY_MARGIN_TOKENS


def test_invalid_budget_is_rejected() -> None:
    """A budget that reserves more than the window would call every prompt admissible."""
    with pytest.raises(BudgetError):
        ContextBudget(max_model_len=100, reserved_completion_tokens=90, safety_margin_tokens=20)
    with pytest.raises(BudgetError):
        ContextBudget(max_model_len=0)


def test_budget_rejects_negative_reserves() -> None:
    with pytest.raises(BudgetError):
        ContextBudget(max_model_len=100, reserved_completion_tokens=-1)
    with pytest.raises(BudgetError):
        ContextBudget(max_model_len=100, safety_margin_tokens=-1)


# ------------------------------------------------------------------ 3-5: measuring


def test_measure_counts_without_special_tokens(tokenizer: FakeTokenizer) -> None:
    """3. Measurement uses add_special_tokens=False, so the count is the prompt's own."""
    assert measure(_messages(7), tokenizer) == 7
    assert tokenizer.calls, "the tokenizer must actually have been called"


def test_measure_requires_a_tokenizer() -> None:
    """4. A missing tokenizer raises rather than returning a guessed count."""
    with pytest.raises(TokenizerUnavailableError):
        measure(_messages(3), None)  # type: ignore[arg-type]


def test_measure_renders_tool_calls(tokenizer: FakeTokenizer) -> None:
    """5. Assistant tool_calls are part of the prompt and must be counted."""
    plain = [{"role": "assistant", "content": "one two three"}]
    with_call = [
        {
            "role": "assistant",
            "content": "one two three",
            "tool_calls": [{"id": "c1", "function": {"name": "read_file", "arguments": "{}"}}],
        }
    ]
    assert measure(with_call, tokenizer) > measure(plain, tokenizer)


def test_measure_flattens_content_parts(tokenizer: FakeTokenizer) -> None:
    """Multi-part content (a list of blocks) must not be dropped from the count."""
    parts = [{"role": "user", "content": [{"type": "text", "text": "a b c d"}]}]
    assert measure(parts, tokenizer) >= 1


# --------------------------------------------------------------- 6-8: preflight


def test_preflight_ok_when_under_capacity(tokenizer: FakeTokenizer, budget: ContextBudget) -> None:
    """6. A prompt inside the budget reports `ok` and the tokens actually left."""
    probe = preflight(_messages(10), tokenizer, budget)
    assert probe.status == STATUS_OK
    assert probe.admissible
    assert probe.prompt_tokens == 10
    assert probe.remaining == budget.capacity - probe.prompt_tokens


def test_preflight_exceeded_at_exact_capacity_plus_one(tokenizer: FakeTokenizer, budget: ContextBudget) -> None:
    """7. The boundary is exact: capacity passes, capacity + 1 does not."""
    at_capacity = preflight(_messages(budget.capacity), tokenizer, budget)
    over = preflight(_messages(budget.capacity + 1), tokenizer, budget)
    assert at_capacity.prompt_tokens == budget.capacity
    assert at_capacity.status == STATUS_OK
    assert over.prompt_tokens == budget.capacity + 1
    assert over.status == STATUS_EXCEEDED
    assert not over.admissible


def test_probe_dict_carries_counts_not_text(tokenizer: FakeTokenizer, budget: ContextBudget) -> None:
    """8. Probe serialisation must never leak prompt text into a row or a log."""
    secret = [{"role": "user", "content": "SUPER_SECRET_MARKER " * 5}]
    probe = preflight(secret, tokenizer, budget)
    payload = json.dumps(probe.to_dict())
    assert "SUPER_SECRET_MARKER" not in payload
    assert isinstance(probe.to_dict()["prompt_tokens"], int)


# ----------------------------------------------------------------- 9-11: compaction


def test_compaction_keeps_system_and_first_user_message(tokenizer: FakeTokenizer) -> None:
    """9. The system prompt and the task instruction are never elided."""
    messages = [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": "TASK INSTRUCTION"},
    ]
    for i in range(10):
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x " * 500})

    result = compact_messages(messages, policy=CompactionPolicy(observation_char_cap=50), tokenizer=tokenizer)
    assert result.messages[0]["content"] == "SYSTEM PROMPT"
    assert result.messages[1]["content"] == "TASK INSTRUCTION"


def test_compaction_preserves_tool_call_id_pairing(tokenizer: FakeTokenizer) -> None:
    """10. Every assistant tool_call keeps its matching tool reply, or the prompt is invalid."""
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(8):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": f"c{i}", "function": {"name": "read_file", "arguments": "{}"}}],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "y " * 400})

    result = compact_messages(messages, policy=CompactionPolicy(keep_recent_tool_messages=2), tokenizer=tokenizer)
    call_ids = {
        call["id"]
        for message in result.messages
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    }
    reply_ids = {m.get("tool_call_id") for m in result.messages if m.get("role") == "tool"}
    assert call_ids == reply_ids, "an orphaned tool reply would be rejected by the server"


def test_compaction_shrinks_and_records_why(tokenizer: FakeTokenizer) -> None:
    """11. Compaction reports what it did, so a shrinking prompt is visible, not silent."""
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(6):
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "z " * 300})

    policy = CompactionPolicy(observation_char_cap=100, keep_recent_tool_messages=1)
    result = compact_messages(messages, policy=policy, tokenizer=tokenizer)
    assert result.compacted
    assert result.chars_after < result.chars_before
    assert result.replaced_observations > 0
    assert result.reasons


def test_compaction_keeps_recent_observations_verbatim(tokenizer: FakeTokenizer) -> None:
    """The model must still see what it just asked for."""
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(5):
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result-{i} " + "q " * 500})

    policy = CompactionPolicy(observation_char_cap=50, keep_recent_tool_messages=2)
    result = compact_messages(messages, policy=policy, tokenizer=tokenizer)
    tails = [m["content"] for m in result.messages if m.get("role") == "tool"][-2:]
    assert all("result-" in tail for tail in tails)


def test_compaction_elision_marker_names_the_loss(tokenizer: FakeTokenizer) -> None:
    """An elided observation says so and says how much was lost."""
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "tool", "tool_call_id": "c0", "content": "LONG " * 2000},
        {"role": "tool", "tool_call_id": "c1", "content": "short"},
    ]
    result = compact_messages(
        messages, policy=CompactionPolicy(observation_char_cap=10, keep_recent_tool_messages=1), tokenizer=tokenizer
    )
    elided = [m["content"] for m in result.messages if m.get("role") == "tool"][0]
    assert "elided" in elided
    assert "LONG LONG" not in elided


# ------------------------------------------------- 12-14: infrastructure failures


def test_infrastructure_failure_never_claims_reward() -> None:
    """12. The object's whole purpose is that a broken run is not a scored run."""
    failure = infrastructure_failure(error_type="context_budget_exceeded", stage="preflight", message="too big")
    assert failure["reward_eligible"] is False
    assert failure["model_executed"] is False
    assert failure["error_type"] == "context_budget_exceeded"
    assert failure["stage"] == "preflight"


def test_infrastructure_failure_carries_the_probe_when_given() -> None:
    """13. The probe travels with the failure, so a diagnosis needs no re-run."""
    budget = ContextBudget(max_model_len=100, reserved_completion_tokens=20, safety_margin_tokens=5)
    probe = Probe(prompt_tokens=90, budget=budget, status=STATUS_EXCEEDED, detail="over")
    failure = infrastructure_failure(error_type="context_budget_exceeded", stage="server", message="400", probe=probe)
    assert failure["context_budget"]["prompt_tokens"] == 90
    assert failure["context_budget"]["status"] == STATUS_EXCEEDED


def test_infrastructure_failure_context_budget_matches_the_schema() -> None:
    """13b. The nested probe record must satisfy the schema, not merely look right.

    Regression: `infrastructure_failure` emitted `Probe.to_dict()`, whose key set is the
    trace's TOP-LEVEL `context_budget` vocabulary. The nested block is a different schema
    object with `additionalProperties: false`, requiring `capacity` and rejecting
    `reserved_completion_tokens` / `safety_margin_tokens` / `context_remaining_tokens`.
    Every persisted infrastructure-failure row therefore failed validation at
    `_persist_evidence`. It stayed latent because a prior run wrote no such row, so the
    path was never exercised - hence asserting schema conformance, not just a key or two.
    """
    schema = json.loads((REPO / "analysis" / "trace_schema.json").read_text(encoding="utf-8"))
    nested = schema["properties"]["infrastructure_failure"]["properties"]["context_budget"]

    budget = ContextBudget(max_model_len=8192, reserved_completion_tokens=1024, safety_margin_tokens=64)
    probe = Probe(prompt_tokens=7200, budget=budget, status=STATUS_EXCEEDED, detail="over budget")
    failure = infrastructure_failure(
        error_type="context_budget_exceeded", stage="preflight", message="over", probe=probe
    )
    record = failure["context_budget"]

    # Exact key set: `additionalProperties: false` makes an extra key a hard failure.
    assert set(record) == {"max_model_len", "capacity", "prompt_tokens", "status", "detail"}
    assert record["capacity"] == 8192 - 1024 - 64
    for required in nested["required"]:
        assert required in record

    # And the real validator agrees, including when the probe measured nothing at all.
    unresolved = infrastructure_failure(
        error_type="tokenizer_unavailable",
        stage="preflight",
        message="no tokenizer",
        probe=Probe(prompt_tokens=None, budget=budget, status=STATUS_TOKENIZER_UNAVAILABLE),
    )
    assert unresolved["context_budget"]["prompt_tokens"] == 0
    assert "detail" not in unresolved["context_budget"]


@pytest.mark.parametrize(
    "row,expected",
    [
        ({"termination_reason": "infrastructure_error"}, True),
        ({"infrastructure_failure": {"reward_eligible": False}}, True),
        (
            {
                "error": {
                    "message": "Error code: 400 - maximum context length is 8192 tokens",
                    "type": "BadRequestError",
                }
            },
            True,
        ),
        ({"termination_reason": "submitted", "result": {"reward": 1.0}}, False),
        ({"error": {"message": "SyntaxError in candidate patch", "type": "ValueError"}}, False),
        ({}, False),
    ],
)
def test_is_infrastructure_failure_detection(row: dict[str, Any], expected: bool) -> None:
    """14. Detection covers the new labels and the legacy 400 message alike."""
    assert is_infrastructure_failure(row) is expected


def test_reward_eligible_rows_drops_only_infrastructure_failures() -> None:
    rows = [
        {"termination_reason": "submitted", "result": {"reward": 1.0}},
        {"termination_reason": "infrastructure_error", "result": {"reward": None}},
        {"termination_reason": "max_turns", "result": {"reward": 0.0}},
    ]
    kept = reward_eligible_rows(rows)
    assert len(kept) == 2
    assert all(row["termination_reason"] != "infrastructure_error" for row in kept)


# ------------------------------------------------------- 15-16: tokenizer loading


def test_load_tokenizer_rejects_a_missing_path() -> None:
    """15. A wrong path is an error, not a silent fallback to another model's vocab."""
    with pytest.raises(TokenizerUnavailableError):
        load_tokenizer(Path("/nonexistent/tokenizer/for/tests"))


def test_load_tokenizer_expands_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The config writes the cache path with `~`, which must be expanded before the stat.

    Without expansion a `~/...` path is a relative directory named "~", so the error
    would say "path does not exist" and point at the wrong problem entirely.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "tokdir").mkdir()
    # The directory now exists, so the failure must move on to the import/load stage
    # rather than reporting a missing path.
    with pytest.raises(TokenizerUnavailableError) as excinfo:
        load_tokenizer("~/tokdir")
    assert "does not exist" not in str(excinfo.value)


def test_tokenizer_unavailable_status_is_distinct() -> None:
    """16. "could not measure" must be distinguishable from "measured and it fits"."""
    assert STATUS_TOKENIZER_UNAVAILABLE != STATUS_OK
    assert STATUS_TOKENIZER_UNAVAILABLE != STATUS_EXCEEDED
    assert issubclass(TokenizerUnavailableError, BudgetError)


# ------------------------------------------------------- 17-18: schema contract


def test_schema_accepts_infrastructure_error_and_the_new_objects() -> None:
    """17. The trace schema must permit what the collector now writes."""
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    props = schema["properties"]
    assert "infrastructure_error" in props["termination_reason"]["enum"]
    assert "infrastructure_failure" in props
    assert "context_budget" in props
    # Optional, never required: an ordinary rollout carries neither.
    assert "infrastructure_failure" not in schema["required"]
    assert "context_budget" not in schema["required"]


def test_schema_keeps_the_row_closed() -> None:
    """The top level stays strictly closed, so the two new keys had to be declared."""
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False


# ------------------------------------------------------------ 19-20: config wiring


def test_config_context_block_is_pinned() -> None:
    """19. Every context knob is present and typed, so a run cannot inherit a guess."""
    import yaml

    config = yaml.safe_load((REPO / "configs" / "eval" / "baseline_phase6.yaml").read_text(encoding="utf-8"))
    context = config["context"]
    assert isinstance(context["max_model_len"], int)
    assert isinstance(context["reserved_completion_tokens"], int)
    assert isinstance(context["safety_margin_tokens"], int)
    assert isinstance(context["observation_char_cap"], int)
    assert isinstance(context["keep_recent_tool_messages"], int)
    # The tokenizer is pinned by revision, exactly as the weights are: a mutable tag
    # would let the probe measure a different vocabulary than the one being served.
    assert "snapshots/2e1fd397ee46e1388853d2af2c993145b0f1098a" in context["tokenizer_path"]
    assert config["context"]["max_model_len"] == config["model"]["max_seq_length"]


def test_config_declares_a_minimum_valid_rollout_count() -> None:
    """20. A batch with a missing task must fail loudly, not shrink the denominator."""
    import yaml

    config = yaml.safe_load((REPO / "configs" / "eval" / "baseline_phase6.yaml").read_text(encoding="utf-8"))
    assert config["run"]["min_valid_rollouts"] > 0
    assert config["run"]["min_valid_rollouts"] <= len(config["run"]["task_subset"])
