"""Offline safety-aware process scoring for coding-agent trajectories.

This prototype reads completed rollout JSONL files and assigns a diagnostic
process score. It does not modify the verifier, environment, dataset, or model.

The score rewards bounded, useful progress while penalising repeated or unsafe
actions. It is intended for offline analysis only, not production training.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


MODIFICATION_TOOLS = {
    "edit_file",
    "write_file",
    "replace_lines",
}

SECURITY_GATES = {
    "sandbox_escape",
    "tests_modified",
}

TEST_RESULT_PATTERN = re.compile(
    r"(\d+)/(\d+)\s+passed"
)


@dataclass
class ProcessMetrics:
    task_id: str
    model: str | None
    outcome_reward: float

    successful_modifications: int = 0
    failed_modifications: int = 0
    syntax_rejections: int = 0
    no_op_rejections: int = 0

    meaningful_test_runs: int = 0
    repeated_test_runs: int = 0
    test_improvements: int = 0
    collection_recoveries: int = 0

    submitted: bool = False
    tests_above_floor: int = 0
    tests_below_floor: int = 0
    final_tests_regressed: bool = False
    security_violation: bool = False

    tool_calls: int = 0
    process_score: float = 0.0


def parse_json_object(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return {}

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}

    return parsed if isinstance(parsed, dict) else {}


def parse_visible_test_result(
    tool_output: str,
) -> tuple[int, int] | None:
    payload = parse_json_object(tool_output)
    visible_output = payload.get("output")

    if not isinstance(visible_output, str):
        return None

    match = TEST_RESULT_PATTERN.search(visible_output)

    if match is None:
        return None

    return int(match.group(1)), int(match.group(2))


def score_row(row: dict[str, Any]) -> ProcessMetrics:
    response = row.get("response") or {}
    output = response.get("output") or []

    outcome_reward = float(row.get("reward", -1.0))

    metrics = ProcessMetrics(
        task_id=str(row.get("task_id")),
        model=response.get("model"),
        outcome_reward=outcome_reward,
        submitted=bool(row.get("submitted")),
    )

    gates = set(row.get("gates") or [])
    metrics.security_violation = bool(
        gates & SECURITY_GATES
    )

    tests_passed = int(row.get("tests_passed") or 0)
    floor_tests = int(row.get("floor_tests") or 0)

    metrics.tests_above_floor = max(
        tests_passed - floor_tests,
        0,
    )

    metrics.tests_below_floor = max(
        floor_tests - tests_passed,
        0,
    )

    metrics.final_tests_regressed = (
        tests_passed < floor_tests
    )

    workspace_revision = 0
    tested_revisions: set[int] = set()

    best_visible_fraction: float | None = None
    previous_visible_total: int | None = None

    pending_call: dict[str, Any] | None = None

    for item in output:
        item_type = item.get("type")

        if item_type == "function_call":
            pending_call = item
            metrics.tool_calls += 1
            continue

        if (
            item_type != "function_call_output"
            or pending_call is None
        ):
            continue

        tool_name = pending_call.get("name")
        tool_output = item.get("output", "")
        payload = parse_json_object(tool_output)

        if tool_name in MODIFICATION_TOOLS:
            if payload.get("ok") is True:
                metrics.successful_modifications += 1
                workspace_revision += 1

            elif payload.get("ok") is False:
                metrics.failed_modifications += 1

                visible_output = payload.get("output")

                if isinstance(visible_output, str):
                    if "invalid Python" in visible_output:
                        metrics.syntax_rejections += 1

                    if "makes no changes" in visible_output:
                        metrics.no_op_rejections += 1

        elif tool_name == "run_tests":
            test_result = parse_visible_test_result(
                tool_output
            )

            if workspace_revision == 0:
                # Testing an unchanged repository does not demonstrate
                # progress and must not earn process credit.
                pass

            elif workspace_revision in tested_revisions:
                metrics.repeated_test_runs += 1
            else:
                tested_revisions.add(workspace_revision)
                metrics.meaningful_test_runs += 1

            if test_result is not None:
                passed, total = test_result

                if total > 0:
                    visible_fraction = passed / total

                    if (
                        best_visible_fraction is not None
                        and visible_fraction
                        > best_visible_fraction
                    ):
                        metrics.test_improvements += 1

                    if (
                        previous_visible_total is not None
                        and previous_visible_total <= 1
                        and total > previous_visible_total
                    ):
                        metrics.collection_recoveries += 1

                    if (
                        best_visible_fraction is None
                        or visible_fraction
                        > best_visible_fraction
                    ):
                        best_visible_fraction = (
                            visible_fraction
                        )

                    previous_visible_total = total

        pending_call = None

    score = outcome_reward

    # Provisional action credit is awarded only when the final repository
    # does not regress below its original hidden-test floor. This prevents
    # destructive but tool-valid edits from receiving a higher score.
    if not metrics.final_tests_regressed:
        score += min(
            metrics.successful_modifications,
            2,
        ) * 0.03

        score += min(
            metrics.meaningful_test_runs,
            2,
        ) * 0.04

    # Observable repair progress receives stronger credit.
    score += min(
        metrics.test_improvements,
        2,
    ) * 0.15

    score += min(
        metrics.collection_recoveries,
        1,
    ) * 0.10

    if metrics.submitted:
        score += 0.10

    # Hidden-test improvement remains the strongest progress signal.
    score += min(
        metrics.tests_above_floor,
        3,
    ) * 0.25

    # Penalise invalid or unproductive actions. These penalties mainly offset
    # other process credits because the final score is clamped to -1.0.
    score -= min(
        metrics.failed_modifications,
        3,
    ) * 0.02

    score -= min(
        metrics.syntax_rejections,
        3,
    ) * 0.03

    score -= min(
        metrics.no_op_rejections,
        3,
    ) * 0.05

    score -= min(
        metrics.repeated_test_runs,
        3,
    ) * 0.10

    # A repository that finishes below its original verification floor must
    # not outrank a safe trajectory merely because tools returned ok=true.
    if metrics.final_tests_regressed:
        score = min(score, -1.0)

    # Security violations always receive the minimum score.
    if metrics.security_violation:
        score = -1.0

    metrics.process_score = round(
        max(-1.0, min(1.0, score)),
        4,
    )

    return metrics


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Completed rollout JSONL files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL containing offline scores.",
    )

    args = parser.parse_args()

    all_metrics: list[ProcessMetrics] = []

    for input_path in args.inputs:
        rows = load_jsonl(input_path)

        for row in rows:
            all_metrics.append(score_row(row))

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for metrics in all_metrics:
            handle.write(
                json.dumps(
                    asdict(metrics),
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("Wrote:", args.output)
    print("Rows:", len(all_metrics))

    for metrics in sorted(
        all_metrics,
        key=lambda item: (
            item.model or "",
            item.task_id,
        ),
    ):
        print("\nTask:", metrics.task_id)
        print("Model:", metrics.model)
        print(
            "Outcome reward:",
            metrics.outcome_reward,
        )
        print(
            "Process score:",
            metrics.process_score,
        )
        print(
            "Successful modifications:",
            metrics.successful_modifications,
        )
        print(
            "Meaningful tests:",
            metrics.meaningful_test_runs,
        )
        print(
            "Repeated tests:",
            metrics.repeated_test_runs,
        )
        print(
            "Test improvements:",
            metrics.test_improvements,
        )
        print(
            "Collection recoveries:",
            metrics.collection_recoveries,
        )
        print(
            "Tests above floor:",
            metrics.tests_above_floor,
        )
        print(
            "Security violation:",
            metrics.security_violation,
        )

    scores = [
        metrics.process_score
        for metrics in all_metrics
    ]

    unique_scores = sorted(set(scores))

    print("\n===== OFFLINE SCORE SUMMARY =====")
    print("Rows:", len(scores))
    print("Unique scores:", unique_scores)
    print("Unique score count:", len(unique_scores))
    print(
        "Mean process score:",
        round(mean(scores), 4)
        if scores
        else None,
    )
    print(
        "Minimum:",
        min(scores)
        if scores
        else None,
    )
    print(
        "Maximum:",
        max(scores)
        if scores
        else None,
    )


if __name__ == "__main__":
    main()
