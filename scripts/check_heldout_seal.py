#!/usr/bin/env python
"""Fail if the sealed held-out split has changed.

The held-out split is the only measurement in this project that cannot be repeated. Once it
has been looked at - to pick a checkpoint, to tune a hyperparameter, to decide a task was
"too hard" - it stops being held out, and every number computed from it afterwards is a
training-set number wearing a held-out label. Nothing about that failure is visible in the
results; the split still validates, the rewards still look reasonable, and the generalisation
claim is quietly worthless.

So the seal is enforced mechanically rather than by discipline. `benchmark/heldout_seal.json`
records the git *tree* hash of `benchmark/tasks/heldout/` at the moment it was sealed. A tree
hash is content-addressed: it covers every byte of every file in the split and nothing else,
so it is unaffected by later commits that touch documentation or reward code, and it changes
the instant anyone edits a task, adds one, or deletes one.

    uv run python scripts/check_heldout_seal.py

Runs as part of `make check`. If it fails, the fix is to restore the split - not to update
the recorded hash.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEAL_PATH = REPO_ROOT / "benchmark" / "heldout_seal.json"
HELDOUT = "benchmark/tasks/heldout"


def current_tree_sha() -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "rev-parse", f"HEAD:{HELDOUT}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"could not read the held-out tree from git:\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def working_tree_is_dirty() -> bool:
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["git", "status", "--porcelain", "--", HELDOUT],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(proc.stdout.strip())


def main() -> int:
    if not SEAL_PATH.is_file():
        print(f"no seal recorded at {SEAL_PATH.relative_to(REPO_ROOT)}; held-out is not sealed")
        return 0

    seal = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
    if not seal.get("sealed"):
        print("held-out is not sealed yet; nothing to check")
        return 0

    expected = seal["heldout_tree_sha"]
    actual = current_tree_sha()

    if actual != expected:
        print(
            f"HELD-OUT SEAL BROKEN\n"
            f"  {HELDOUT}/ has changed since it was sealed on {seal['sealed_on']}.\n"
            f"    sealed tree: {expected}\n"
            f"    current tree: {actual}\n\n"
            f"  Restore the split. Do NOT edit benchmark/heldout_seal.json to make this pass -\n"
            f"  the recorded hash is the evidence that the held-out numbers were measured once,\n"
            f"  on a split nobody had tuned against.\n\n"
            f"  See docs/heldout_seal.md.",
            file=sys.stderr,
        )
        return 1

    if working_tree_is_dirty():
        print(
            f"HELD-OUT SEAL BROKEN\n"
            f"  {HELDOUT}/ has uncommitted modifications. The committed tree still matches the\n"
            f"  seal, but the working copy does not - so a graded run right now would not be\n"
            f"  scoring the sealed split. Revert them:\n"
            f"    git checkout -- {HELDOUT}",
            file=sys.stderr,
        )
        return 1

    print(f"held-out sealed {seal['sealed_on']}: {seal['n_tasks']} tasks, tree {expected[:12]} unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
