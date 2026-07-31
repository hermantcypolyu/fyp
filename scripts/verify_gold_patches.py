#!/usr/bin/env python
"""Prove every task is both solvable and failable, and report the reward floor.

For each task this runs the hidden suite twice:

  * **untouched repo**  - must fail at least one test (AGENT_BUILD_SPEC 5.4 rule 2).
    The fraction that still passes is the task's **reward floor**: what an agent scores
    for doing nothing at all.
  * **gold patch applied** - must pass every test (5.4 rule 1). If the reference
    solution cannot reach 1.0, the task's ceiling is below 1 and every number computed
    from it is wrong.

The gap between floor and 1.0 is the headroom the RL signal has to work with. A task
with a 0.95 floor is technically valid and practically useless, so the summary sorts by
headroom and flags the thin ones.

    make verify-gold
    uv run python scripts/verify_gold_patches.py --split train
    uv run python scripts/verify_gold_patches.py --task bugfix_offbyone_pagination --keep
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.tools.patch_utils import PatchError  # noqa: E402
from benchmark.tools.task_loader import SPLITS, TASKS_ROOT, Task, TaskError, grade, iter_tasks, load_task  # noqa: E402

#: Below this much room between floor and 1.0, the task barely discriminates.
THIN_HEADROOM = 0.25


def check(task: Task, keep_dir: Path | None) -> dict:
    row: dict = {
        "id": task.id,
        "split": task.split,
        "category": task.category,
        "difficulty": task.difficulty,
        "n_tests": task.n_tests,
    }
    try:
        base = grade(task, apply_gold=False, work_dir=(keep_dir / "base") if keep_dir else None)
        gold = grade(task, apply_gold=True, work_dir=(keep_dir / "gold") if keep_dir else None)
    except (TaskError, PatchError) as e:
        row.update(ok=False, error=str(e))
        return row

    row.update(
        collected=base.collected,
        floor=round(base.pass_fraction, 4),
        base_summary=base.summary(),
        gold_summary=gold.summary(),
        failable=not base.all_passed and not base.collection_error,
        solvable=gold.all_passed,
        headroom=round(1.0 - base.pass_fraction, 4),
    )
    row["ok"] = bool(row["failable"] and row["solvable"])
    if not row["ok"]:
        row["tail"] = (gold if not gold.all_passed else base).stdout_tail[-1200:]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--task", help="Check a single task by id.")
    parser.add_argument(
        "--keep",
        metavar="DIR",
        nargs="?",
        const="results/raw/gold_workdirs",
        help="Keep the materialised working copies here for inspection.",
    )
    parser.add_argument("--json", metavar="PATH", help="Write the full report as JSON.")
    args = parser.parse_args()

    if args.task:
        found = [TASKS_ROOT / s / args.task for s in SPLITS if (TASKS_ROOT / s / args.task).is_dir()]
        if not found:
            print(f"No task directory named '{args.task}'", file=sys.stderr)
            return 2
        tasks = [load_task(found[0])]
    else:
        tasks = list(iter_tasks(SPLITS if args.split == "all" else (args.split,)))

    if not tasks:
        print("No tasks found.")
        return 0

    print(f"{'task':<38} {'cat':<14} {'diff':<7} {'untouched':<12} {'+gold':<12} {'floor':<7} ok")
    print("-" * 100)

    rows = []
    for task in tasks:
        keep = Path(args.keep) / task.id if args.keep else None
        if keep:
            keep.mkdir(parents=True, exist_ok=True)
        row = check(task, keep)
        rows.append(row)
        if "error" in row:
            print(f"{task.id:<38} {task.category:<14} {task.difficulty:<7} {'ERROR':<12} {'-':<12} {'-':<7} no")
            print(f"    {row['error']}")
            continue
        print(
            f"{row['id']:<38} {row['category']:<14} {row['difficulty']:<7} "
            f"{row['base_summary']:<12} {row['gold_summary']:<12} {row['floor']:<7.2f} "
            f"{'yes' if row['ok'] else 'NO'}"
        )
        if not row["ok"]:
            reason = []
            if not row.get("failable"):
                reason.append("untouched repo passes everything (not failable)")
            if not row.get("solvable"):
                reason.append("gold patch does not reach 1.0 (not solvable)")
            print(f"    {'; '.join(reason)}")
            if row.get("tail"):
                for line in row["tail"].strip().splitlines()[-10:]:
                    print(f"    | {line}")

    ok_rows = [r for r in rows if r.get("ok")]
    bad = [r for r in rows if not r.get("ok")]
    thin = [r for r in ok_rows if r["headroom"] < THIN_HEADROOM]

    print("-" * 100)
    if ok_rows:
        floors = [r["floor"] for r in ok_rows]
        print(f"reward floor: min {min(floors):.2f} / mean {sum(floors) / len(floors):.2f} / max {max(floors):.2f}")
    for row in thin:
        print(
            f"  thin headroom: {row['id']} scores {row['floor']:.2f} with no work at all "
            f"(only {row['headroom']:.2f} of range left for the agent)"
        )

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"), "tasks": rows},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {out}")

    if bad:
        print(f"\nFAILED: {len(bad)}/{len(rows)} task(s) are not both solvable and failable")
        return 1
    print(f"\nPASSED: {len(rows)}/{len(rows)} task(s) solvable by gold and failable without it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
