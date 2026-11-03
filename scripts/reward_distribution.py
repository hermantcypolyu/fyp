"""Score existing rollouts under `reward_v1` and `reward_v2` and compare the two distributions.

This answers one question before any GPU time is spent: **does adding a progress component
actually create variance where v1 had none?** The whole premise of Phase 3 is that a group
sitting on the floor is a zero-variance group. If v2 also produces zero variance on the same
data, the component is not doing its job and the weight should not reach training.

What this can and cannot measure today
--------------------------------------
An earlier version of this script claimed the per-turn test counts were simply not on disk
and reported `progress = None` for every row. That was wrong. `run_tests` already writes its
summary into `ToolResult.output` as `"N/M passed (smoke tests only; ...)"`, and the rollout
JSONL keeps the whole tool transcript, so `smoke_delta` recovers progress from every row
without touching the environment.

Doing so gives a result more interesting than the gap it replaced: **95 of the 162 rows ran
the smoke suite, and not one of them ever improved.** The component is correct; the behaviour
it prices never occurs. `--detail` prints that breakdown.

The script therefore reports:

1. the v1 distribution against the v2 distribution, per run and pooled - real, measured now;
2. the measured progress availability, and how often progress was actually positive;
3. the group-variance table - how many groups v1 sees as degenerate, and what v2's progress
   component could add to each.

Number 3 is a ceiling, not a result. It says: "for these N groups v1 has zero spread, and
the progress component can supply at most this much" - it does not say it will.

Usage::

    uv run python scripts/reward_distribution.py
    uv run python scripts/reward_distribution.py --json results/analysis/reward_v2_offline.json
    uv run python scripts/reward_distribution.py --detail
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rewards.compose import GATE_REWARD
from rewards.v2 import PROGRESS_WEIGHT, compose_v2, progress_reward

__all__ = [
    "RunStats",
    "group_variance_table",
    "load_rollouts",
    "main",
    "score_run",
    "smoke_counts_of",
    "smoke_delta_of",
]

RAW = Path("results/raw")

#: The summary line `run_tests` prints. Duplicated from `rewards.v2` because this script
#: also needs the *total*, which `smoke_delta` deliberately discards.
_PASSED_OF = re.compile(r"(\d+)\s*/\s*(\d+)\s+passed")


@dataclass(frozen=True)
class RunStats:
    run: str
    rows: int
    v1_mean: float
    v1_std: float
    v2_mean: float
    v2_std: float

    @property
    def v1_degenerate(self) -> bool:
        return self.rows > 1 and self.v1_std == 0.0


def load_rollouts(root: Path = RAW) -> list[dict[str, Any]]:
    """Every non-empty `rollouts.jsonl` row under `root`, tagged with its run directory."""
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/rollouts.jsonl")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["_run"] = path.parent.name
            rows.append(row)
    return rows


def _v1_of(row: dict[str, Any]) -> float | None:
    """The reward already recorded on the row, if it is the v1 scale."""
    value = row.get("reward")
    return float(value) if isinstance(value, int | float) else None


def smoke_counts_of(row: dict[str, Any]) -> list[int]:
    """Passing-test counts from every `run_tests` call in one rollout, in order.

    The transcript keeps one entry per tool call under `response.output`, each with an
    `output` string; `run_tests` puts `"N/M passed (...)"` at the front of it. Reading the
    transcript rather than a stored counter is what makes progress recoverable from rollouts
    produced *before* this component existed.
    """
    counts: list[int] = []
    for entry in row.get("response", {}).get("output", []) or []:
        match = _PASSED_OF.search(entry.get("output") or "")
        if match:
            counts.append(int(match.group(1)))
    return counts


def smoke_delta_of(row: dict[str, Any]) -> int | None:
    """Best-seen improvement in passing tests, or `None` when no `run_tests` call was recorded.

    Mirrors `rewards.v2.smoke_delta` but reads the whole row: the baseline is the *first*
    reading and the result is the best one, floored at 0. Undoing a good edit is not negative
    progress, because component A already prices the damage in the final patch.
    """
    counts = smoke_counts_of(row)
    if not counts:
        return None
    return max(0, max(counts) - counts[0])


def _progress_of(row: dict[str, Any]) -> float | None:
    """Component B for one historical row: `None` if the smoke suite never ran."""
    return progress_reward(smoke_delta_of(row))


def _v2_of(row: dict[str, Any]) -> float | None:
    """Recompute under v2, with progress recovered from the tool transcript where possible."""
    if "tests_passed" not in row or "tests_total" not in row:
        return None
    floor = int(row.get("floor_tests") or 0)
    passed = int(row["tests_passed"])
    total = int(row["tests_total"])
    gates = tuple(row.get("gates") or ())
    if total <= floor:
        # Historical rows where collection failed, so `tests_total` is 0 or below the floor.
        # Every one of them carries a `test_count_mismatch` gate, so the score is decided by
        # the gate and the denominator never matters. If a row ever reached here *without* a
        # gate, the recorded counts would contradict task.yaml and v2 must not invent a number.
        if gates:
            return GATE_REWARD
        return None
    return compose_v2(passed, total, floor, gates=gates, progress=_progress_of(row)).final


def score_run(rows: Sequence[dict[str, Any]]) -> RunStats:
    v1 = [v for v in (_v1_of(r) for r in rows) if v is not None]
    v2 = [v for v in (_v2_of(r) for r in rows) if v is not None]
    return RunStats(
        run=str(rows[0].get("_run", "?")),
        rows=len(rows),
        v1_mean=statistics.fmean(v1) if v1 else float("nan"),
        v1_std=statistics.pstdev(v1) if len(v1) > 1 else 0.0,
        v2_mean=statistics.fmean(v2) if v2 else float("nan"),
        v2_std=statistics.pstdev(v2) if len(v2) > 1 else 0.0,
    )


def group_variance_table(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (run, task) group: v1's spread, and whether v2's progress term could rescue it.

    A GRPO group is the set of rollouts sharing a prompt; v1's spread over that set is the
    only thing that produces a gradient. `max_progress_std` is the largest `final` standard
    deviation the progress component could add to that group:

        sigma_final = PROGRESS_WEIGHT * sigma_progress,  sigma_progress <= 0.5
                                                    ->  sigma_final <= 0.10

    so it is 0.10 for any degenerate non-gated group, and 0.0 for a group where every member
    is gated (a gate sets `final` to `GATE_REWARD` regardless of progress).
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if _v1_of(row) is not None:
            groups[(str(row.get("_run", "?")), str(row.get("task_id", "?")))].append(row)

    table = []
    for (run, task_id), members in sorted(groups.items()):
        rewards = [r for r in (_v1_of(m) for m in members) if r is not None]
        std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
        all_gated = all((m.get("gates") or ()) for m in members)
        table.append(
            {
                "run": run,
                "task_id": task_id,
                "n": len(rewards),
                "v1_mean": round(statistics.fmean(rewards), 4),
                "v1_std": round(std, 4),
                "degenerate": len(rewards) > 1 and std == 0.0,
                "all_gated": all_gated,
                # The ceiling on what progress can add, not a requirement it will meet.
                "max_progress_std": 0.0 if all_gated else round(PROGRESS_WEIGHT * 0.5, 4),
            }
        )
    return table


def progress_availability(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """How often component B had anything to measure, and how often it fired.

    This is the measurement that matters more than the weight. A progress component that is
    correct but observes zero progress on every rollout is not a broken reward; it is a
    statement that the edit-then-retest loop the reward prices never happens.
    """
    rows = list(rows)
    deltas = [d for d in (smoke_delta_of(r) for r in rows) if d is not None]
    multi = sum(1 for r in rows if len(smoke_counts_of(r)) > 1)
    return {
        "rows": len(rows),
        "rows_with_a_test_run": len(deltas),
        "rows_with_positive_progress": sum(1 for d in deltas if d > 0),
        "rows_with_more_than_one_test_run": multi,
        "max_observed_delta": max(deltas) if deltas else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW, help="directory of run subdirectories")
    parser.add_argument("--json", type=Path, default=None, help="write the full report here")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="list every rollout together with its recovered progress delta",
    )
    args = parser.parse_args(argv)

    rows = load_rollouts(args.raw)
    if not rows:
        print(f"no rollouts found under {args.raw}")
        return 1

    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_run[str(row["_run"])].append(row)

    stats = [score_run(group) for group in by_run.values() if len(group) > 1]
    stats.sort(key=lambda s: (s.run,))

    print(f"{'run':44s} {'rows':>4s} {'v1 mean':>8s} {'v1 std':>7s} {'v2 mean':>8s} {'v2 std':>7s}")
    print("-" * 84)
    for s in stats:
        flag = "  <- v1 degenerate" if s.v1_degenerate else ""
        print(f"{s.run:44s} {s.rows:4d} {s.v1_mean:8.3f} {s.v1_std:7.3f} {s.v2_mean:8.3f} {s.v2_std:7.3f}{flag}")

    pooled_v1 = [v for v in (_v1_of(r) for r in rows) if v is not None]
    pooled_v2 = [v for v in (_v2_of(r) for r in rows) if v is not None]
    print("-" * 84)
    print(
        f"{'pooled':44s} {len(rows):4d} "
        f"{statistics.fmean(pooled_v1):8.3f} {statistics.pstdev(pooled_v1) if len(pooled_v1) > 1 else 0.0:7.3f} "
        f"{statistics.fmean(pooled_v2):8.3f} {statistics.pstdev(pooled_v2) if len(pooled_v2) > 1 else 0.0:7.3f}"
    )

    table = group_variance_table(rows)
    degenerate = [g for g in table if g["degenerate"]]
    multi = [g for g in table if g["n"] > 1]
    rescuable = [g for g in degenerate if not g["all_gated"]]
    gated_out = [g for g in degenerate if g["all_gated"]]

    availability = progress_availability(rows)
    print()
    print("progress availability (component B):")
    print(f"  rows on disk:                       {availability['rows']}")
    print(f"  rows that ran the smoke suite:      {availability['rows_with_a_test_run']}")
    print(f"  rows that ran it more than once:    {availability['rows_with_more_than_one_test_run']}")
    print(f"  rows with positive progress:        {availability['rows_with_positive_progress']}")
    print(f"  largest observed delta:             {availability['max_observed_delta']}")
    if availability["rows_with_positive_progress"] == 0:
        print()
        print("  B is correct and measurable, but it observed no improvement anywhere. The")
        print("  reward cannot price behaviour that never happens; this is a model-behaviour")
        print("  finding, not an instrumentation gap.")

    if args.detail:
        print()
        print(f"{'run':44s} {'task':34s} {'counts':28s} {'delta':>5s}")
        print("-" * 116)
        for row in rows:
            counts = smoke_counts_of(row)
            if not counts:
                continue
            seq = ",".join(str(c) for c in counts[:6]) + ("..." if len(counts) > 6 else "")
            print(
                f"{str(row.get('_run', '?')):44s} {str(row.get('task_id', '?')):34s} {seq:28s} {smoke_delta_of(row):5d}"
            )

    print()
    print(f"groups (run x task) with >1 rollout: {len(multi)}")
    print(f"of those, zero v1 spread:           {len(degenerate)}")
    print(f"  - progress could add up to 0.10:  {len(rescuable)}")
    print(f"  - all members gated (stays 0):    {len(gated_out)}")
    if rescuable:
        print()
        print("These contribute no gradient under reward_v1. Progress can add at most 0.10 of")
        print("spread to each. Note that ceiling is unreachable while B reads 0 everywhere -")
        print("the measured reality is that these groups stay degenerate:")
        print()
        for g in rescuable[:20]:
            print(f"  {g['run']:40s} {g['task_id']:38s} n={g['n']}")
    if gated_out:
        print()
        print("These have no gradient and cannot gain one from the reward: every member is")
        print("gated, so every member scores GATE_REWARD whatever its components say. The fix")
        print("is rollouts that differ, not weights:")
        print()
        for g in gated_out[:20]:
            print(f"  {g['run']:40s} {g['task_id']:38s} n={g['n']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "rows": len(rows),
                    "runs": len(by_run),
                    "progress_availability": availability,
                    "per_run": [s.__dict__ for s in stats],
                    "pooled": {
                        "v1_mean": statistics.fmean(pooled_v1) if pooled_v1 else None,
                        "v1_std": statistics.pstdev(pooled_v1) if len(pooled_v1) > 1 else 0.0,
                        "v2_mean": statistics.fmean(pooled_v2) if pooled_v2 else None,
                        "v2_std": statistics.pstdev(pooled_v2) if len(pooled_v2) > 1 else 0.0,
                    },
                    "groups": table,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
