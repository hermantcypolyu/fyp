"""Phase 6K: peak-probe telemetry and the compaction-attempt counter.

Phase 6J recorded `peak_prompt_tokens: 703` and `status: "ok"` for a rollout that was
terminated because its prompt reached 8154 tokens on turn 7. The field was not lying
about turn 0; it was only ever *fed* turn 0. A telemetry field that reports the best
moment of a run answers a different question from the one it appears to answer.

These tests pin the two corrections:

1. `_peak_probe` keeps the maximum, so "how close did this rollout get to the wall" is
   answered by the closest approach rather than the first measurement;
2. `compaction_attempts` is reported alongside `compaction_events`, so "never needed to
   compact" and "compacted repeatedly and still could not fit" are distinguishable.

The schema assertions go through `build_row` with `validate=True` - the same write path
the collector uses - rather than a hand-built dict. Spot-checking keys is how the Phase 6F
telemetry defect survived: the record was well-formed until a closed schema rejected it,
and no clean run ever wrote the field.
"""

from __future__ import annotations

from typing import Any

import pytest

from analysis.context_budget import ContextBudget, Probe
from analysis.trace import Timer, TraceError, build_row, validate_row
from benchmark.tools.task_loader import load_task
from envs.coding_env.verifier import VerifyResult
from scripts.collect_baseline_rollouts import _peak_probe, _public_context_telemetry

from .test_context_budget import REPO

TASK = REPO / "benchmark" / "tasks" / "train" / "bugfix_dotenv_bom_prefix"


def _budget() -> ContextBudget:
    return ContextBudget(max_model_len=8192, reserved_completion_tokens=1024, safety_margin_tokens=64)


def _probe(prompt_tokens: int | None, *, status: str = "ok") -> Probe:
    return Probe(budget=_budget(), prompt_tokens=prompt_tokens, status=status)


def _validated_row(telemetry: dict[str, Any] | None) -> dict[str, Any]:
    """A real v3 row with `telemetry` attached, validated by the collector's own path."""
    task = load_task(TASK)
    result = VerifyResult(
        task_id=task.id,
        reward=0.0,
        pass_fraction=0.0,
        passed=0,
        collected=0,
        submitted=False,
        limits_enforced={"wall_clock_s": True, "network": True},
    )
    row = build_row(
        run_id="phase6k-unit-probe",
        rollout_id="phase6k-unit-probe__bugfix_offbyone_pagination__r00",
        task=task,
        result=result,
        tool_calls=[],
        timer=Timer.start(),
        validate=True,
    )
    # `context_budget` is written by the collector after `build_row` returns, so the field
    # is validated explicitly once the telemetry is in place.
    row["context_budget"] = telemetry
    validate_row(row)
    return row


# ------------------------------------------------------------------ the peak rule
def test_peak_probe_keeps_the_larger_measurement() -> None:
    """Turn 7 at 8154 tokens outranks turn 0 at 703, whichever order they arrive in."""
    early = _probe(703)
    late = _probe(8154, status="context_budget_exceeded")

    assert _peak_probe(early, late) is late
    assert _peak_probe(late, early) is late, "the rule is order-independent"


def test_peak_probe_ignores_an_unmeasured_candidate() -> None:
    """A probe that could not measure does not displace one that did."""
    measured = _probe(703)
    unmeasured = _probe(None, status="tokenizer_unavailable")
    assert _peak_probe(measured, unmeasured) is measured


def test_peak_probe_reports_the_peak_status_not_the_first() -> None:
    """`status` follows the peak: the record answers "how close did it get"."""
    peak = _peak_probe(_probe(703), _probe(8154, status="context_budget_exceeded"))
    assert peak.status == "context_budget_exceeded"

    telemetry = _public_context_telemetry(peak, 3, compaction_attempts=4)
    assert telemetry is not None
    assert telemetry["peak_prompt_tokens"] == 8154
    assert telemetry["status"] == "context_budget_exceeded"


# ------------------------------------------------------- the attempt/event pair
def test_telemetry_distinguishes_attempts_from_events() -> None:
    """Three attempts that removed nothing is a different story from three that did."""
    failures = _public_context_telemetry(_probe(703), 0, compaction_attempts=3)
    assert failures is not None
    assert failures["compaction_attempts"] == 3
    assert failures["compaction_events"] == 0

    successes = _public_context_telemetry(_probe(703), 3, compaction_attempts=3)
    assert successes is not None
    assert successes["compaction_events"] == 3


def test_attempt_counter_is_optional() -> None:
    """Omitting the argument leaves the key out, so older callers keep the old shape."""
    telemetry = _public_context_telemetry(_probe(703), 1)
    assert telemetry is not None
    assert "compaction_attempts" not in telemetry


# ------------------------------------------------------------- schema is the judge
def test_telemetry_with_attempts_validates_against_the_real_schema() -> None:
    """The record the collector now writes must pass the closed schema unchanged."""
    telemetry = _public_context_telemetry(_probe(8154, status="context_budget_exceeded"), 3, compaction_attempts=4)
    row = _validated_row(telemetry)
    assert row["context_budget"]["compaction_attempts"] == 4


def test_telemetry_without_attempts_validates_against_the_real_schema() -> None:
    """The additive field is optional, so the pre-existing shape still validates."""
    telemetry = _public_context_telemetry(_probe(703), 1)
    row = _validated_row(telemetry)
    assert "compaction_attempts" not in row["context_budget"]


def test_negative_attempt_count_is_rejected_by_the_schema() -> None:
    """The schema states `minimum: 0`; a negative counter is a bug, not a value."""
    telemetry = _public_context_telemetry(_probe(703), 0, compaction_attempts=-1)
    with pytest.raises(TraceError):
        _validated_row(telemetry)


def test_an_undeclared_telemetry_key_is_rejected() -> None:
    """Guards the guard: the object is closed, so a stray key cannot slip through."""
    telemetry = _public_context_telemetry(_probe(703), 0, compaction_attempts=0)
    assert telemetry is not None
    telemetry["invented_field"] = 1
    with pytest.raises(TraceError):
        _validated_row(telemetry)
