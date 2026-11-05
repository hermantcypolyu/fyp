"""Tests for the Phase 6.3 analysis.

The metrics in `scripts/baseline_analysis.py` are the numbers the whole project will be
judged by, so they are checked against hand-computed fixtures rather than only against
each other. Each fixture is small enough that the expected value can be worked out by
hand and written next to the assertion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.baseline_analysis import (  # noqa: E402
    AnalysisError,
    analyse,
    failure_category,
    is_syntax_failure,
    load_rows,
)

# --------------------------------------------------------------------------- fixtures


def _row(
    *,
    task_id: str = "task_a",
    reward: float,
    submitted: bool = True,
    turns: int = 5,
    tool_calls: int = 4,
    tokens: int = 100,
    duration: float = 10.0,
    gates: list[str] | None = None,
    tests: tuple[int, int] = (1, 1),
    hidden: tuple[int, int] = (1, 1),
    termination: str = "submitted",
    error: dict | None = None,
    steps: list[dict] | None = None,
) -> dict[str, Any]:
    passed, total = tests
    hidden_passed, hidden_declared = hidden
    return {
        "schema_version": "3",
        "run_id": "fixture",
        "task": {"id": task_id},
        "result": {
            "reward": reward,
            "submitted": submitted,
            "gates": gates or [],
            "tests_passed": passed,
            "tests_total": total,
        },
        "hidden_tests": {"passed": hidden_passed, "declared": hidden_declared},
        "turns": {"turns_used": turns},
        "budgets": {"tool_calls_used": tool_calls},
        "token_usage": {"total": tokens},
        "duration_s": duration,
        "termination_reason": termination,
        "error": error,
        "steps": steps or [],
    }


# ----------------------------------------------------------------------- core metrics


def test_metrics_match_a_hand_computed_fixture() -> None:
    """Two solved and two failed rollouts, with round numbers so the arithmetic is exact."""
    rows = [
        _row(reward=1.0, turns=4, tool_calls=3, tokens=100, duration=10.0, hidden=(2, 2)),
        _row(reward=1.0, turns=6, tool_calls=5, tokens=200, duration=20.0, hidden=(2, 2)),
        _row(reward=-1.0, submitted=False, turns=8, tool_calls=7, tokens=300, duration=30.0, hidden=(0, 2)),
        _row(reward=0.0, turns=2, tool_calls=1, tokens=400, duration=40.0, hidden=(0, 2)),
    ]
    m = analyse(rows)

    # rewards (1, 1, -1, 0) -> sum 1, mean 0.25; population stdev = sqrt(mean((x-0.25)^2))
    # deviations: .75, .75, -1.25, -.25 -> squares .5625,.5625,1.5625,.0625 -> mean .6875
    assert m["reward"]["mean"] == pytest.approx(0.25)
    assert m["reward"]["stdev"] == pytest.approx(0.6875**0.5)
    assert m["reward"]["min"] == -1.0
    assert m["reward"]["max"] == 1.0
    assert m["reward"]["distinct"] == [-1.0, 0.0, 1.0]
    assert m["reward"]["distinct_count"] == 3
    assert m["reward"]["has_variance"] is True

    assert m["pass_at_1"] == pytest.approx(0.5)  # 2 of 4
    assert m["valid_submission_rate"] == pytest.approx(0.75)  # 3 of 4
    assert m["hidden_test_success_rate"] == pytest.approx(0.5)  # 4 passed of 8 declared
    assert m["hidden_tests_passed"] == 4
    assert m["hidden_tests_declared"] == 8

    assert m["mean_turns"] == pytest.approx(5.0)  # (4+6+8+2)/4
    assert m["mean_tool_calls"] == pytest.approx(4.0)  # (3+5+7+1)/4
    assert m["mean_token_usage"] == pytest.approx(250.0)  # (100+200+300+400)/4
    assert m["mean_runtime_s"] == pytest.approx(25.0)  # (10+20+30+40)/4


def test_zero_denominator_rates_are_null_not_zero() -> None:
    """'we did not measure' and 'we measured zero' are different claims."""
    rows = [_row(reward=1.0, hidden=(0, 0))]
    m = analyse(rows)
    assert m["hidden_test_success_rate"] is None


def test_single_sample_has_no_stdev_claim() -> None:
    """Stdev of one sample is undefined; 0.0 would falsely assert zero variance."""
    m = analyse([_row(reward=1.0)])
    assert m["reward"]["stdev"] is None


def test_uniform_rewards_report_no_variance() -> None:
    m = analyse([_row(reward=-1.0), _row(reward=-1.0)])
    assert m["reward"]["distinct_count"] == 1
    assert m["reward"]["has_variance"] is False


# ---------------------------------------------------------------------- per-task split


def test_per_task_metrics_are_computed_per_task() -> None:
    rows = [
        _row(task_id="task_a", reward=1.0, turns=2),
        _row(task_id="task_a", reward=-1.0, turns=4),
        _row(task_id="task_b", reward=-1.0, turns=10),
    ]
    m = analyse(rows)
    assert m["n_tasks"] == 2
    assert m["per_task"]["task_a"]["n_rollouts"] == 2
    assert m["per_task"]["task_a"]["pass_at_1"] == pytest.approx(0.5)
    assert m["per_task"]["task_a"]["mean_turns"] == pytest.approx(3.0)
    assert m["per_task"]["task_b"]["pass_at_1"] == pytest.approx(0.0)
    # Global mean turns is (2+4+10)/3 = 16/3, not the mean of the two per-task means (5.0).
    assert m["mean_turns"] == pytest.approx(16 / 3)


# ------------------------------------------------------------------------- categorising


def test_solved_rows_are_categorised_as_success() -> None:
    assert failure_category(_row(reward=1.0)) == "success"


def test_no_submission_is_not_reported_as_a_test_failure() -> None:
    """A run that never submitted did not 'fail the tests' - it failed to finish."""
    row = _row(reward=-1.0, submitted=False, termination="max_turns")
    assert failure_category(row) == "no_submission_turns_exhausted"

    row = _row(reward=-1.0, submitted=False, termination="max_tool_calls")
    assert failure_category(row) == "no_submission_tool_budget"


def test_safety_gates_outrank_test_failure() -> None:
    """'tried to escape the sandbox' is a more important label than 'wrong answer'."""
    row = _row(reward=-1.0, gates=["sandbox_escape", "no_submit"])
    assert failure_category(row) == "safety_gate_sandbox_escape"

    row = _row(reward=-1.0, gates=["tests_modified"])
    assert failure_category(row) == "safety_gate_tests_modified"


def test_syntax_failure_detected_from_step_observation() -> None:
    row = _row(
        reward=-1.0,
        submitted=False,
        steps=[{"ok": False, "observation_summary": "Error: invalid syntax at line 7"}],
    )
    assert is_syntax_failure(row) is True
    assert failure_category(row) == "syntax_failure"


def test_syntax_failure_detected_from_harness_error() -> None:
    row = _row(reward=-1.0, error={"type": "SyntaxError", "message": "unexpected EOF"})
    assert is_syntax_failure(row) is True


def test_successful_steps_do_not_count_as_syntax_failures() -> None:
    row = _row(
        reward=1.0,
        steps=[{"ok": True, "observation_summary": "invalid syntax"}],
    )
    assert is_syntax_failure(row) is False


def test_harness_error_is_its_own_category() -> None:
    row = _row(reward=-1.0, error={"type": "ConnectionError", "message": "boom"})
    assert failure_category(row) == "harness_error"


def test_partial_tests_passed_is_distinguished_from_all_failed() -> None:
    partial = _row(reward=0.0, submitted=True, tests=(3, 11))
    assert failure_category(partial) == "partial_tests_passed"
    failed = _row(reward=-1.0, submitted=True, tests=(0, 11))
    assert failure_category(failed) == "all_tests_failed"


# ------------------------------------------------------------------------------ loading


def test_loading_rejects_a_wrong_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    bad = _row(reward=0.0)
    bad["schema_version"] = "2"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(AnalysisError, match="schema_version"):
        load_rows(path)


def test_loading_an_empty_file_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(AnalysisError, match="no rollouts"):
        load_rows(path)


def test_loading_a_missing_file_says_what_to_run(tmp_path: Path) -> None:
    with pytest.raises(AnalysisError, match="collect_baseline_rollouts"):
        load_rows(tmp_path / "absent.jsonl")
