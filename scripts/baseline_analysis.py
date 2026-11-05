"""Phase 6.3 - baseline analysis over the collected rollout trace.

Reads `results/raw/phase6_baseline_1p5b/rollouts.jsonl` (v3 trace rows) and computes every
metric Phase 6.3 asks for, writing them to `results/analysis/phase6_baseline_1p5b.json`.

Two rules shape this file:

  * Every metric is computed from the trace, never from a re-run. The trace is the artifact
    of record; if a metric cannot be derived from it, the metric gets `null` rather than a
    plausible-looking substitute.
  * Missing data is reported as missing. Rates are `null` when the denominator is zero, not
    `0.0` - an absent measurement and a measured zero are different claims.

`reward` here is the *live* verifier reward (`reward_v1`), because that is what the row
records. `reward_v2` is analysed separately (see `rewards/v2.py`) and is not mixed in.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

log = logging.getLogger("baseline_analysis")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUTS = REPO_ROOT / "results" / "raw" / "phase6_baseline_1p5b" / "rollouts.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "analysis" / "phase6_baseline_1p5b.json"


class AnalysisError(RuntimeError):
    """The trace cannot support the requested analysis."""


# --------------------------------------------------------------------------- loading


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AnalysisError(f"{path}: no rollouts file. Run scripts/collect_baseline_rollouts.py first.")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise AnalysisError(f"{path}:{lineno}: not valid JSON: {e}") from e
            if row.get("schema_version") != "3":
                raise AnalysisError(f"{path}:{lineno}: schema_version is {row.get('schema_version')!r}, expected '3'")
            rows.append(row)
    if not rows:
        raise AnalysisError(f"{path}: contains no rollouts")
    return rows


# --------------------------------------------------------------------------- helpers


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate, or `None` when the denominator is zero.

    Returning 0.0 for an empty denominator would claim "we measured zero", which is a
    different - and false - statement from "we did not measure".
    """
    if denominator == 0:
        return None
    return numerator / denominator


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _stdev(values: list[float]) -> float | None:
    """Population standard deviation; `None` for a single sample.

    Sample stdev of one value is undefined, and 0.0 would wrongly assert "no variance was
    observed" when in fact variance could not be observed.
    """
    if len(values) < 2:
        return None
    return statistics.pstdev(values)


def _reward(row: dict[str, Any]) -> float:
    return float(row["result"]["reward"])


def is_infrastructure_failure(row: dict[str, Any]) -> bool:
    """Did this rollout fail for a reason that is not the policy's fault?

    Three signals, any sufficient: the Phase 6B `infrastructure_failure` object, the
    `infrastructure_error` termination, or - for rows collected before Phase 6B existed -
    a model-server 400 that names the context window. That last case is why this check
    has to read the message: the 48 Phase 6 rows were written by a collector that had no
    way to label an overflow, so without this they look like policy failures and quietly
    drag every mean down.

    Such a row never played an episode the policy could be judged on, so including it in
    an aggregate would be scoring the harness.
    """
    if row.get("termination_reason") == "infrastructure_error":
        return True
    failure = row.get("infrastructure_failure")
    if isinstance(failure, dict) and failure.get("reward_eligible") is False:
        return True
    message = str((row.get("error") or {}).get("message") or "")
    return "maximum context length" in message and "400" in message


def reward_eligible_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows an aggregate may be computed over."""
    return [row for row in rows if not is_infrastructure_failure(row)]


def _steps(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row.get("steps") or []


def _turns(row: dict[str, Any]) -> int:
    return int(row.get("turns", {}).get("turns_used", 0))


def _tokens(row: dict[str, Any]) -> int:
    usage = row.get("token_usage") or {}
    return int(usage.get("total") or 0)


def _duration(row: dict[str, Any]) -> float:
    return float(row.get("duration_s") or 0.0)


def _gates(row: dict[str, Any]) -> list[str]:
    return list(row.get("result", {}).get("gates") or [])


def _tests(row: dict[str, Any]) -> tuple[int, int]:
    result = row.get("result", {})
    return int(result.get("tests_passed") or 0), int(result.get("tests_total") or 0)


# --------------------------------------------------------------------------- failures


SYNTAX_MARKERS = (
    "SyntaxError",
    "IndentationError",
    "TabError",
    "invalid syntax",
    "unexpected EOF",
)

# Gates that mean the sandbox caught something unsafe or dishonest, as opposed to merely
# scoring badly. These are reported as a separate rate because "tried to escape" is not
# the same failure as "produced the wrong answer".
SAFETY_GATES = ("sandbox_escape", "tests_modified")


def is_syntax_failure(row: dict[str, Any]) -> bool:
    """Did the episode produce code that does not parse?

    Two signals, either sufficient: an `ok: false` step whose observation carries a parser
    message, or a graded result that reports zero passing tests against a real test run
    with a syntax error in the transcript.
    """
    for step in _steps(row):
        if step.get("ok"):
            continue
        observation = str(step.get("observation_summary") or "")
        if any(marker in observation for marker in SYNTAX_MARKERS):
            return True
    error = row.get("error") or {}
    message = str(error.get("message") or "")
    return any(marker in message for marker in SYNTAX_MARKERS)


def failure_category(row: dict[str, Any]) -> str:
    """One label per failed rollout, chosen by the first matching rule.

    Order matters: a run that never submitted did not "fail the tests", it failed to
    finish. Reporting it as a test failure would overstate how close the model got.
    """
    if is_infrastructure_failure(row):
        # Checked first: an infrastructure failure has no reward to compare against and
        # must never be filed as a policy failure such as `harness_error`.
        return "infrastructure_failure"
    if _reward(row) == 1.0:
        return "success"
    error = row.get("error") or {}
    if error:
        return "harness_error"
    if "sandbox_escape" in _gates(row):
        return "safety_gate_sandbox_escape"
    if "tests_modified" in _gates(row):
        return "safety_gate_tests_modified"
    if is_syntax_failure(row):
        return "syntax_failure"
    if not row.get("result", {}).get("submitted"):
        termination = row.get("termination_reason")
        if termination == "max_tool_calls":
            return "no_submission_tool_budget"
        if termination == "timeout":
            return "no_submission_timeout"
        return "no_submission_turns_exhausted"
    passed, total = _tests(row)
    if total == 0:
        return "no_tests_recorded"
    if passed == 0:
        return "all_tests_failed"
    return "partial_tests_passed"


# --------------------------------------------------------------------------- metrics


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Every rate and mean below is computed over reward-eligible rows only. The full row
    # list still drives the failure categories and the termination counts, because those
    # are precisely where an infrastructure failure has to remain visible.
    eligible = reward_eligible_rows(rows)
    rewards = [_reward(r) for r in eligible]
    turns = [_turns(r) for r in eligible]
    calls = [int(r.get("budgets", {}).get("tool_calls_used") or 0) for r in eligible]
    tokens = [_tokens(r) for r in eligible]
    durations = [_duration(r) for r in eligible]

    submitted = [r for r in eligible if r.get("result", {}).get("submitted")]
    solved = [r for r in eligible if _reward(r) == 1.0]

    # Hidden-test success: the fraction of *hidden* tests passed, which is not the same as
    # pass_fraction on a partial submission - a run that fixed one of eleven tests still
    # has to have submitted for those passes to count.
    hidden_passed = 0
    hidden_declared = 0
    for row in eligible:
        hidden = row.get("hidden_tests") or {}
        hidden_passed += int(hidden.get("passed") or 0)
        hidden_declared += int(hidden.get("declared") or 0)

    safety_hits = [r for r in eligible if any(gate in _gates(r) for gate in SAFETY_GATES)]
    syntax = [r for r in eligible if is_syntax_failure(r)]

    categories = Counter(failure_category(r) for r in rows)

    by_task: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"rewards": [], "turns": [], "tool_calls": [], "tokens": [], "duration_s": []}
    )
    for row in eligible:
        task_id = row["task"]["id"]
        bucket = by_task[task_id]
        bucket["rewards"].append(_reward(row))
        bucket["turns"].append(_turns(row))
        bucket["tool_calls"].append(int(row.get("budgets", {}).get("tool_calls_used") or 0))
        bucket["tokens"].append(_tokens(row))
        bucket["duration_s"].append(_duration(row))

    per_task = {
        task_id: {
            "n_rollouts": len(bucket["rewards"]),
            "mean_reward": _mean(bucket["rewards"]),
            "reward_stdev": _stdev(bucket["rewards"]),
            "distinct_rewards": sorted(set(bucket["rewards"])),
            "pass_at_1": _rate(sum(1 for r in bucket["rewards"] if r == 1.0), len(bucket["rewards"])),
            "mean_turns": _mean([float(t) for t in bucket["turns"]]),
            "mean_tool_calls": _mean([float(c) for c in bucket["tool_calls"]]),
            "mean_tokens": _mean([float(t) for t in bucket["tokens"]]),
            "mean_duration_s": _mean(bucket["duration_s"]),
        }
        for task_id, bucket in sorted(by_task.items())
    }

    distinct = sorted(set(rewards))
    return {
        "run_id": rows[0].get("run_id"),
        "schema_version": rows[0].get("schema_version"),
        "n_rollouts": len(rows),
        "n_reward_eligible": len(eligible),
        "n_infrastructure_invalid": len(rows) - len(eligible),
        "n_tasks": len({r["task"]["id"] for r in rows}),
        "tasks_with_zero_valid_rollouts": sorted({r["task"]["id"] for r in rows} - set(by_task)),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "reward": {
            # `min`/`max` are None rather than raising when nothing was eligible: an
            # empty aggregate is a finding, not a crash.
            "mean": _mean(rewards),
            "stdev": _stdev(rewards),
            "min": min(rewards) if rewards else None,
            "max": max(rewards) if rewards else None,
            "distinct": distinct,
            # The 6.2 requirement, restated as a number the report can be checked against.
            "distinct_count": len(distinct),
            "has_variance": len(distinct) > 1,
        },
        "pass_at_1": _rate(len(solved), len(eligible)),
        "hidden_test_success_rate": _rate(hidden_passed, hidden_declared),
        "hidden_tests_passed": hidden_passed,
        "hidden_tests_declared": hidden_declared,
        "valid_submission_rate": _rate(len(submitted), len(eligible)),
        "syntax_failure_rate": _rate(len(syntax), len(eligible)),
        "safety_gate_rate": _rate(len(safety_hits), len(eligible)),
        "mean_turns": _mean([float(t) for t in turns]),
        "mean_tool_calls": _mean([float(c) for c in calls]),
        "mean_token_usage": _mean([float(t) for t in tokens]),
        "mean_runtime_s": _mean(durations),
        "failure_categories": dict(sorted(categories.items())),
        "termination_reasons": dict(sorted(Counter(r.get("termination_reason") for r in rows).items())),
        "per_task": per_task,
    }


# --------------------------------------------------------------------------- reporting


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(result: dict[str, Any]) -> str:
    lines = [
        f"Phase 6.3 baseline analysis - {result['run_id']}",
        f"  rollouts: {result['n_rollouts']}  tasks: {result['n_tasks']}  errors: {result['n_errors']}",
        (
            f"  reward-eligible: {result['n_reward_eligible']}  "
            f"infrastructure-invalid: {result['n_infrastructure_invalid']}"
        ),
        "",
        "  mean reward          " + _fmt(result["reward"]["mean"]),
        "  reward stdev         " + _fmt(result["reward"]["stdev"]),
        "  distinct rewards     " + str(result["reward"]["distinct"]),
        "  pass@1               " + _fmt(result["pass_at_1"]),
        "  hidden-test success  " + _fmt(result["hidden_test_success_rate"]),
        "  valid submissions    " + _fmt(result["valid_submission_rate"]),
        "  syntax failures      " + _fmt(result["syntax_failure_rate"]),
        "  safety gates         " + _fmt(result["safety_gate_rate"]),
        "  mean turns           " + _fmt(result["mean_turns"], 1),
        "  mean tool calls      " + _fmt(result["mean_tool_calls"], 1),
        "  mean tokens          " + _fmt(result["mean_token_usage"], 0),
        "  mean runtime (s)     " + _fmt(result["mean_runtime_s"], 1),
        "",
        "  failure categories:",
    ]
    for name, count in result["failure_categories"].items():
        lines.append(f"    {name:32s} {count}")
    lines.append("")
    zero = result.get("tasks_with_zero_valid_rollouts") or []
    if zero:
        # Printed as a warning rather than folded into the per-task table: a task absent
        # from that table has no mean to display, and its absence is the finding.
        lines.append("  WARNING - tasks with zero reward-eligible rollouts:")
        for task_id in zero:
            lines.append(f"    {task_id}")
        lines.append("")
    lines.append("  per task:")
    for task_id, m in result["per_task"].items():
        lines.append(
            f"    {task_id:34s} n={m['n_rollouts']:<3d} mean={_fmt(m['mean_reward'])} "
            f"pass@1={_fmt(m['pass_at_1'])} turns={_fmt(m['mean_turns'], 1)} "
            f"calls={_fmt(m['mean_tool_calls'], 1)}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        rows = load_rows(args.rollouts)
        result = analyse(rows)
    except AnalysisError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render(result))
        print(f"\n  written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
