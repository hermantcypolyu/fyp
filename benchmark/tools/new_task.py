#!/usr/bin/env python
"""Scaffold a blank task directory.

The point of this script is that the *only* creative work left after running it is
writing the broken repo, the hidden tests and the fixed files. Everything structural -
the directory layout, the mandatory forbidden globs, the per-category defaults, the
placeholder that makes the validator complain until you replace it - comes out
pre-filled and already in the shape `validate_tasks.py` expects.

    make new-task ID=bugfix_retry_backoff CATEGORY=bug_fix
    uv run python benchmark/tools/new_task.py bugfix_retry_backoff --category bug_fix \
        --split heldout --difficulty medium

Then, in order:

  1. write the broken code under `repo/`
  2. write the hidden suite under `tests/`
  3. copy the files you intend to change into `solution/fixed/`, with the fix applied
  4. `make gold-patches`
  5. `make validate && make verify-gold`

See docs/task_authoring_guide.md for the long version.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.tools.task_loader import SPLITS, TASKS_ROOT  # noqa: E402

CATEGORIES = ("bug_fix", "test_update", "config_fix", "small_feature", "docs_improve")
DIFFICULTIES = ("easy", "medium", "hard")
ID_RE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")

#: Per-category starting points. These are defaults a human should still review, not law.
DEFAULTS: dict[str, dict] = {
    "bug_fix": {
        "entrypoint": "src/<module>.py",
        "allowed": ['"repo/src/**"', '"repo/*.py"'],
        "extra_forbidden": [],
        "files_changed": 1,
        "turns": 20,
        "calls": 40,
        "timeout": 60,
    },
    "test_update": {
        # The agent edits the repo's *own* visible suite. That lives at repo/tests/**,
        # which is a different path from the hidden suite at tests/**.
        "entrypoint": "tests/test_<module>.py",
        "allowed": ['"repo/tests/**"'],
        "extra_forbidden": ['"repo/src/**"'],
        "files_changed": 1,
        "turns": 25,
        "calls": 50,
        "timeout": 180,
    },
    "config_fix": {
        "entrypoint": "pyproject.toml",
        "allowed": ['"repo/pyproject.toml"', '"repo/*.toml"', '"repo/*.cfg"', '"repo/*.md"'],
        "extra_forbidden": [],
        "files_changed": 2,
        "turns": 25,
        "calls": 50,
        "timeout": 60,
    },
    "small_feature": {
        "entrypoint": "src/<module>.py",
        "allowed": ['"repo/src/**"', '"repo/*.py"', '"repo/*.md"'],
        "extra_forbidden": [],
        "files_changed": 2,
        "turns": 30,
        "calls": 60,
        "timeout": 60,
    },
    "docs_improve": {
        "entrypoint": "README.md",
        "allowed": ['"repo/*.md"', '"repo/docs/**"', '"repo/src/**"'],
        "extra_forbidden": [],
        "files_changed": 1,
        "turns": 20,
        "calls": 40,
        "timeout": 60,
    },
}

#: Deliberately left in the generated file. The validator fails on the LOC placeholder and
#: on a missing gold patch, so a half-finished task can never quietly enter the corpus.
TODO = "TODO"


def task_yaml(task_id: str, category: str, difficulty: str, split: str) -> str:
    d = DEFAULTS[category]
    allowed = "\n".join(f"    - {g}" for g in d["allowed"])
    extra_forbidden = "".join(f"\n    - {g}" for g in d["extra_forbidden"])
    return f"""\
id: {task_id}
category: {category}
difficulty: {difficulty}
split: {split}

# What the user would say. Describe the SYMPTOM, never the hidden tests: no test file
# names, no assertion counts, no "make test_foo pass". A competent developer should be
# able to fix this from the description alone.
instruction: >
  {TODO} - describe the observable problem in 3-6 sentences.

repo:
  root: repo/
  # Run `make validate`; it measures repo/ for you and tells you the number to put here.
  loc: 0
  entrypoint_hint: {d["entrypoint"]}

verification:
  runner: pytest
  test_files: [tests/test_{TODO.lower()}.py]
  timeout_s: {d["timeout"]}
  n_tests: 0

constraints:
  allowed_write_globs:
{allowed}
  forbidden_write_globs:
    - "tests/**"
    - "solution/**"
    - "**/conftest.py"
    - "task.yaml"
    - "repo/smoke/**"{extra_forbidden}
  max_turns: {d["turns"]}
  max_tool_calls: {d["calls"]}

provenance:
  # real_repo | textbook_pattern | synthetic
  # real_repo additionally requires `origin: <url> @ <commit-sha>` and a `notes` entry
  # saying exactly which commit was inverted. Target is >=50% real_repo across the corpus.
  source: textbook_pattern
  license: MIT
  notes: >
    {TODO} - where this came from and why it is representative.

metadata:
  expected_files_changed: {d["files_changed"]}
  created: {date.today().isoformat()}
  tags: []
"""


README_STUB = """\
# {task_id}

{TODO}: a short, honest README for the *fictional* project the agent is working on.
Write it as if the agent had just been handed this repository at work. It may describe
intended behaviour - that is a legitimate source of information - but it must not point
at the hidden tests.
"""

HIDDEN_STUB = '''\
"""Hidden verification for `{task_id}`.

The agent never sees this file: `materialise(..., include_tests=False)` simply does not
write it into the sandbox. Rules:

  * Deterministic. No clocks, no network, no randomness, no filesystem outside the copy.
  * Between 3 and 40 tests, and every one of them independently meaningful - reward_v1 is
    the pass fraction, so each test is worth 1/n of the whole signal.
  * Fine-grained enough that a partly-correct answer scores between 0 and 1. A single
    all-or-nothing test destroys the gradient the whole project depends on.
  * The untouched repo must fail at least one of them; the gold patch must pass all.
"""

import pytest  # noqa: F401  - remove if unused


def test_{TODO_lower}():
    raise AssertionError("write the hidden tests")
'''

SMOKE_STUB = '''\
"""Visible smoke checks - this is what `run_tests()` shows the agent.

Keep these WEAK and keep them passing in both the broken and the fixed state. They exist
so the agent can confirm the code still imports, not so it can read its own grade. If a
smoke test is byte-identical to a hidden test the validator rejects the task.
"""


def test_module_imports():
    raise AssertionError("write a shallow smoke test, or delete the smoke/ directory")
'''

FIXED_README = """\
Put the *fixed* copies of the files you intend to change in here, mirroring their paths
under `repo/`. For example, if the fix touches `repo/src/pagination.py`, the corrected
file goes at `solution/fixed/src/pagination.py`.

Then run `make gold-patches`, which diffs this tree against `repo/` and writes
`solution/gold.patch`. Never hand-write the patch: the generator normalises line endings,
which is the single most common reason a hand-rolled diff fails to apply on Windows.

This directory is never materialised into the agent's sandbox.
"""


def build(task_id: str, category: str, difficulty: str, split: str, force: bool) -> Path:
    dest = TASKS_ROOT / split / task_id
    if dest.exists() and not force:
        raise SystemExit(f"{dest} already exists (pass --force to overwrite the scaffold)")

    for sub in ("repo/src", "tests", "smoke", "solution/fixed"):
        (dest / sub).mkdir(parents=True, exist_ok=True)

    files = {
        "task.yaml": task_yaml(task_id, category, difficulty, split),
        "repo/README.md": README_STUB.format(task_id=task_id, TODO=TODO),
        "repo/src/__init__.py": '"""TODO: the package under test."""\n',
        f"tests/test_{TODO.lower()}.py": HIDDEN_STUB.format(task_id=task_id, TODO_lower=TODO.lower()),
        "smoke/test_smoke.py": SMOKE_STUB,
        "solution/fixed/README.md": FIXED_README,
    }
    if category == "test_update":
        (dest / "repo/tests").mkdir(parents=True, exist_ok=True)
        files["repo/tests/test_placeholder.py"] = (
            '"""The repo\'s OWN visible suite - this is what the agent edits.\n\n'
            "Distinct from the hidden suite one level up. Write it in the state a real\n"
            'stale suite would be in: calling the old API, failing to import.\n"""\n'
        )

    for rel, content in files.items():
        path = dest / rel
        if path.exists() and not force:
            continue
        path.write_text(content, encoding="utf-8")

    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("id", help="snake_case task id; also the directory name")
    parser.add_argument("--category", required=True, choices=CATEGORIES)
    parser.add_argument("--difficulty", default="medium", choices=DIFFICULTIES)
    parser.add_argument("--split", default="train", choices=SPLITS)
    parser.add_argument("--force", action="store_true", help="overwrite an existing scaffold")
    args = parser.parse_args()

    if not ID_RE.match(args.id):
        print(f"'{args.id}' is not a valid id: use lowercase words joined by underscores.", file=sys.stderr)
        return 2

    dest = build(args.id, args.category, args.difficulty, args.split, args.force)
    rel = dest.relative_to(REPO_ROOT)

    print(f"scaffolded {rel}\n")
    print("next:")
    print(f"  1. write the broken code under      {rel / 'repo'}")
    print(f"  2. write the hidden suite under     {rel / 'tests'}")
    print(f"  3. put fixed files under            {rel / 'solution' / 'fixed'}")
    print("  4. make gold-patches")
    print("  5. make validate && make verify-gold")
    print("\ndocs/task_authoring_guide.md has the full checklist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
