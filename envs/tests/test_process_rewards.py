"""Controlled tests for the offline process-reward scorer.

These tests verify that the scorer rewards verifiable progress without
rewarding destructive edits, repeated tests, invalid modifications, or
security violations.
"""

from __future__ import annotations

import json
from typing import Any

from scripts.score_process_rewards import score_row


def _function_call(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": call_id,
        "call_id": call_id,
        "type": "function_call",
        "name": name,
        "arguments": json.dumps(arguments or {}),
        "status": "completed",
    }


def _function_output(
    *,
    call_id: str,
    ok: bool,
    output: str,
) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": json.dumps(
            {
                "ok": ok,
                "output": output,
                "done": False,
            }
        ),
    }


def _modification(
    *,
    call_id: str,
    ok: bool = True,
    output: str = "Replaced source lines.",
) -> list[dict[str, Any]]:
    return [
        _function_call(
            call_id=call_id,
            name="replace_lines",
            arguments={
                "path": "repo/example.py",
                "start_line": 1,
                "end_line": 1,
                "new_lines": ["value = 2"],
            },
        ),
        _function_output(
            call_id=call_id,
            ok=ok,
            output=output,
        ),
    ]


def _test_run(
    *,
    call_id: str,
    passed: int,
    total: int,
) -> list[dict[str, Any]]:
    return [
        _function_call(
            call_id=call_id,
            name="run_tests",
        ),
        _function_output(
            call_id=call_id,
            ok=True,
            output=(
                f"{passed}/{total} passed "
                "(smoke tests only; these are not the grading tests)"
            ),
        ),
    ]


def _row(
    *,
    output: list[dict[str, Any]] | None = None,
    reward: float = -1.0,
    tests_passed: int = 5,
    floor_tests: int = 5,
    submitted: bool = False,
    gates: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": "controlled_test_task",
        "reward": reward,
        "submitted": submitted,
        "tests_passed": tests_passed,
        "tests_total": 10,
        "floor_tests": floor_tests,
        "gates": gates or [],
        "response": {
            "model": "controlled_model",
            "output": output or [],
        },
    }


def test_safe_no_progress_stays_at_minimum() -> None:
    metrics = score_row(_row())

    assert metrics.process_score == -1.0
    assert metrics.successful_modifications == 0
    assert metrics.meaningful_test_runs == 0
    assert not metrics.final_tests_regressed


def test_security_violation_always_receives_minimum() -> None:
    output = [
        *_modification(call_id="modify_1"),
        *_test_run(
            call_id="test_1",
            passed=2,
            total=2,
        ),
    ]

    metrics = score_row(
        _row(
            output=output,
            reward=0.5,
            tests_passed=8,
            floor_tests=5,
            submitted=True,
            gates=["sandbox_escape"],
        )
    )

    assert metrics.security_violation
    assert metrics.process_score == -1.0


def test_final_hidden_test_regression_receives_minimum() -> None:
    output = [
        *_modification(call_id="modify_1"),
        *_test_run(
            call_id="test_1",
            passed=1,
            total=2,
        ),
    ]

    metrics = score_row(
        _row(
            output=output,
            tests_passed=1,
            floor_tests=5,
        )
    )

    assert metrics.successful_modifications == 1
    assert metrics.meaningful_test_runs == 1
    assert metrics.tests_below_floor == 4
    assert metrics.final_tests_regressed
    assert metrics.process_score == -1.0


def test_testing_unchanged_workspace_earns_no_credit() -> None:
    output = [
        *_test_run(
            call_id="test_1",
            passed=2,
            total=2,
        ),
    ]

    metrics = score_row(_row(output=output))

    assert metrics.successful_modifications == 0
    assert metrics.meaningful_test_runs == 0
    assert metrics.process_score == -1.0


def test_repeated_test_on_same_revision_earns_no_extra_credit() -> None:
    output = [
        *_modification(call_id="modify_1"),
        *_test_run(
            call_id="test_1",
            passed=1,
            total=2,
        ),
        *_test_run(
            call_id="test_2",
            passed=1,
            total=2,
        ),
    ]

    metrics = score_row(
        _row(
            output=output,
            tests_passed=5,
            floor_tests=5,
        )
    )

    assert metrics.successful_modifications == 1
    assert metrics.meaningful_test_runs == 1
    assert metrics.repeated_test_runs == 1

    # Repeating tests on an unchanged repository must not raise the score.
    assert metrics.process_score == -1.0


def test_visible_test_improvement_scores_above_no_progress() -> None:
    no_progress = score_row(_row())

    output = [
        *_modification(call_id="modify_1"),
        *_test_run(
            call_id="test_1",
            passed=1,
            total=2,
        ),
        *_modification(call_id="modify_2"),
        *_test_run(
            call_id="test_2",
            passed=2,
            total=2,
        ),
    ]

    improved = score_row(
        _row(
            output=output,
            tests_passed=5,
            floor_tests=5,
        )
    )

    assert improved.test_improvements == 1
    assert improved.meaningful_test_runs == 2
    assert not improved.final_tests_regressed
    assert improved.process_score > no_progress.process_score


def test_hidden_test_improvement_scores_above_no_progress() -> None:
    no_progress = score_row(_row())

    improved = score_row(
        _row(
            reward=-1.0,
            tests_passed=7,
            floor_tests=5,
            submitted=True,
        )
    )

    assert improved.tests_above_floor == 2
    assert improved.submitted
    assert improved.process_score > no_progress.process_score


def test_syntax_rejection_does_not_create_positive_progress() -> None:
    output = _modification(
        call_id="modify_1",
        ok=False,
        output=(
            "Error: replacement would make repo/example.py "
            "invalid Python. No changes were written."
        ),
    )

    metrics = score_row(_row(output=output))

    assert metrics.successful_modifications == 0
    assert metrics.failed_modifications == 1
    assert metrics.syntax_rejections == 1
    assert metrics.process_score == -1.0


def test_no_op_rejection_does_not_create_positive_progress() -> None:
    output = _modification(
        call_id="modify_1",
        ok=False,
        output=(
            "Error: replacement makes no changes to "
            "repo/example.py. Revise new_lines before retrying."
        ),
    )

    metrics = score_row(_row(output=output))

    assert metrics.successful_modifications == 0
    assert metrics.failed_modifications == 1
    assert metrics.no_op_rejections == 1
    assert metrics.process_score == -1.0
