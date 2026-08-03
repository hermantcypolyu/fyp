#!/usr/bin/env python
"""Validate benchmark tasks against task.schema.json plus the rules the schema can't express.

A JSON schema can check that `n_tests` is an integer. It cannot check that the integer
matches the number of tests pytest actually collects. Most of the value is in the second
kind of check, so most of this file is those.

Every failure prints a code, what is wrong, and what to do about it. If you ever find
yourself reading a message here and thinking "but what do I actually change?", that is a
bug in this file - fix the message.

    make validate                       # everything
    uv run python benchmark/schema/validate_tasks.py --split train
    uv run python benchmark/schema/validate_tasks.py --task bugfix_offbyone_pagination
    uv run python benchmark/schema/validate_tasks.py --all --require-mix   # B2 gate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator  # noqa: E402

from benchmark.tools.globs import normalise, path_matches_any  # noqa: E402
from benchmark.tools.patch_utils import PatchError, patched_files  # noqa: E402
from benchmark.tools.task_loader import (  # noqa: E402
    SPLITS,
    TASKS_ROOT,
    Task,
    TaskError,
    count_loc,
    grade,
    iter_tasks,
    load_task,
)

SCHEMA_PATH = Path(__file__).parent / "task.schema.json"

#: AGENT_BUILD_SPEC 5.2. Enforced only under --require-mix, so partial batches validate cleanly.
REQUIRED_MIX: dict[str, dict[str, int]] = {
    "train": {"bug_fix": 10, "test_update": 6, "config_fix": 5, "small_feature": 6, "docs_improve": 3},
    "heldout": {"bug_fix": 4, "test_update": 2, "config_fix": 1, "small_feature": 2, "docs_improve": 1},
}
TARGET_DIFFICULTY = {"easy": 0.40, "medium": 0.40, "hard": 0.20}
DIFFICULTY_TOLERANCE = 0.15
MIN_REAL_REPO_FRACTION = 0.50
MANDATORY_FORBIDDEN_GLOBS = ("tests/**", "solution/**")
PERMISSIVE_LICENCES = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "PSF-2.0", "CC0-1.0", "Unlicense", "ISC"}


@dataclass
class Finding:
    code: str
    task: str
    message: str
    fix: str = ""
    level: str = "error"  # error | warn

    def render(self) -> str:
        mark = "FAIL" if self.level == "error" else "WARN"
        out = f"  [{mark} {self.code}] {self.message}"
        if self.fix:
            out += f"\n         -> {self.fix}"
        return out


class Report:
    def __init__(self) -> None:
        self.findings: list[Finding] = []
        self.notes: list[str] = []

    def error(self, code: str, task: str, message: str, fix: str = "") -> None:
        self.findings.append(Finding(code, task, message, fix, "error"))

    def warn(self, code: str, task: str, message: str, fix: str = "") -> None:
        self.findings.append(Finding(code, task, message, fix, "warn"))

    def note(self, message: str) -> None:
        """Something --fix-counts changed on disk. Not a finding; the author must still see it."""
        self.notes.append(message)

    def for_task(self, task_id: str) -> list[Finding]:
        return [f for f in self.findings if f.task == task_id]

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warn"]


# ------------------------------------------------------------------ per-task checks


def check_schema(task: Task, report: Report, validator: Draft202012Validator) -> bool:
    errors = sorted(validator.iter_errors(task.data), key=lambda e: list(e.absolute_path))
    for err in errors:
        where = ".".join(str(p) for p in err.absolute_path) or "<root>"
        report.error("SCHEMA", task.id, f"{where}: {err.message}", _schema_hint(err))
    return not errors


def _schema_hint(err) -> str:
    validator = err.validator
    if validator == "required":
        return "Add the missing key. See AGENT_BUILD_SPEC 5.1 for the full contract."
    if validator == "enum":
        return f"Allowed values: {', '.join(map(str, err.validator_value))}"
    if validator == "additionalProperties":
        return "Unknown key - check the spelling, or add it to task.schema.json with an ADR."
    if validator in ("minimum", "maximum"):
        return f"Bound is {err.validator_value}."
    if validator == "pattern":
        return f"Must match /{err.validator_value}/"
    return ""


def check_layout(task: Task, report: Report) -> None:
    tid = task.id

    if task.id != task.path.name:
        report.error(
            "ID_DIR",
            tid,
            f"id '{task.id}' != directory name '{task.path.name}'",
            f"Rename the directory to '{task.id}', or change id to '{task.path.name}'. "
            "ids appear in result files, so prefer renaming the directory.",
        )
    if task.split != task.path.parent.name:
        report.error(
            "SPLIT_DIR",
            tid,
            f"split '{task.split}' != parent directory '{task.path.parent.name}'",
            f"Set split: {task.path.parent.name}, or move the directory.",
        )

    if not task.repo_dir.is_dir():
        report.error("NO_REPO", tid, "missing repo/", "The agent needs something to edit. Create repo/.")
    elif not any(p.suffix == ".py" for p in task.repo_dir.rglob("*")):
        report.error("EMPTY_REPO", tid, "repo/ contains no .py files", "Add the source the task is about.")

    if not task.tests_dir.is_dir():
        report.error("NO_TESTS", tid, "missing tests/", "Hidden tests live here. Create tests/.")

    for rel in task.test_files:
        if not (task.path / rel).is_file():
            report.error(
                "TEST_MISSING",
                tid,
                f"verification.test_files references '{rel}', which does not exist",
                f"Create {task.path.name}/{rel}, or correct the path in task.yaml.",
            )

    if not task.gold_patch_path.is_file():
        report.error(
            "NO_GOLD",
            tid,
            "missing solution/gold.patch",
            "Generate it: uv run python benchmark/tools/make_gold_patch.py --task " + tid,
        )

    for junk in list(task.path.rglob("__pycache__")) + list(task.path.rglob("*.pyc")):
        report.error(
            "JUNK",
            tid,
            f"build artefact committed: {junk.relative_to(task.path)}",
            "Delete it and keep it out of git.",
        )

    hint = task.data.get("repo", {}).get("entrypoint_hint")
    if hint and not (task.repo_dir / hint).exists():
        report.error(
            "HINT_MISSING",
            tid,
            f"repo.entrypoint_hint '{hint}' does not exist under repo/",
            "Point it at a real file, or drop the key (hard tasks should omit it).",
        )


def check_loc(task: Task, report: Report, *, fix: bool = False) -> None:
    if not task.repo_dir.is_dir():
        return
    actual = count_loc(task.repo_dir)
    declared = task.data.get("repo", {}).get("loc")
    if declared != actual:
        if fix and rewrite_scalar(task, r"^(\s*loc:\s*)\d+\s*$", actual):
            report.note(f"{task.id}: repo.loc {declared} -> {actual}")
        else:
            report.error(
                "LOC",
                task.id,
                f"repo.loc is {declared}, but repo/ actually has {actual} lines of code",
                f"Set repo.loc: {actual}",
            )
    if actual > 300:
        report.warn(
            "LOC_BIG",
            task.id,
            f"repo/ is {actual} LOC; 5.4 rule 6 prefers under 300",
            "Trim it. Long repos burn context and, later, VRAM.",
        )


def rewrite_scalar(task: Task, pattern: str, value: int) -> bool:
    """Rewrite one `key: <int>` line in the task's YAML in place.

    Only ever used for fields that are a *measurement* rather than a judgement - the LOC
    of `repo/` and the number of tests pytest collects. Everything else an author has to
    mean on purpose. A regex rather than a YAML round-trip because dumping the parsed
    document would reflow every block scalar in the file and destroy the prose.
    """
    path = task.path / "task.yaml"
    text = path.read_text(encoding="utf-8")
    new_text, n = re.subn(pattern, lambda m: f"{m.group(1)}{value}", text, count=1, flags=re.MULTILINE)
    if n != 1:
        return False
    path.write_text(new_text, encoding="utf-8", newline="\n")
    task.data["repo"] = task.data.get("repo", {})
    return True


def check_constraints(task: Task, report: Report) -> None:
    constraints = task.data.get("constraints", {})
    forbidden = constraints.get("forbidden_write_globs", [])
    allowed = constraints.get("allowed_write_globs", [])
    tid = task.id

    for required in MANDATORY_FORBIDDEN_GLOBS:
        if required not in forbidden:
            report.error(
                "FORBID_MISSING",
                tid,
                f"forbidden_write_globs is missing '{required}'",
                "This is the reward-hacking surface from 5.7. It is mandatory on every task.",
            )

    for pattern in allowed:
        if path_matches_any("tests/test_x.py", [pattern]) or path_matches_any("solution/gold.patch", [pattern]):
            report.error(
                "ALLOW_TOO_WIDE",
                tid,
                f"allowed_write_globs entry '{pattern}' reaches tests/ or solution/",
                "Narrow it. Forbidden wins over allowed at runtime, but a glob this wide means the "
                "task file is lying about intent.",
            )

    turns = constraints.get("max_turns", 0)
    calls = constraints.get("max_tool_calls", 0)
    if calls < turns:
        report.error(
            "BUDGET",
            tid,
            f"max_tool_calls ({calls}) < max_turns ({turns})",
            "One turn can emit several tool calls, so the call budget must be the larger of the two.",
        )


def check_gold_patch(task: Task, report: Report) -> None:
    if not task.gold_patch_path.is_file():
        return
    tid = task.id
    try:
        files = patched_files(task.gold_patch())
    except PatchError as e:
        report.error("GOLD_PARSE", tid, f"solution/gold.patch does not parse: {e}", "Regenerate it, do not hand-edit.")
        return

    if not files:
        report.error(
            "GOLD_EMPTY", tid, "solution/gold.patch touches no files", "An empty gold patch cannot solve anything."
        )
        return

    constraints = task.data.get("constraints", {})
    allowed, forbidden = constraints.get("allowed_write_globs", []), constraints.get("forbidden_write_globs", [])
    for path in files:
        if path_matches_any(path, forbidden):
            report.error(
                "GOLD_FORBIDDEN",
                tid,
                f"gold patch writes '{path}', which forbidden_write_globs blocks",
                "The gold solution must be reachable by a well-behaved agent. Fix the patch or the globs.",
            )
        elif not path_matches_any(path, allowed):
            report.error(
                "GOLD_NOT_ALLOWED",
                tid,
                f"gold patch writes '{path}', which allowed_write_globs does not cover",
                f"Add a glob covering '{path}' - otherwise the task is unsolvable within its own rules.",
            )

    expected = task.data.get("metadata", {}).get("expected_files_changed")
    if expected != len(files):
        report.error(
            "FILES_CHANGED",
            tid,
            f"metadata.expected_files_changed is {expected}, gold patch touches {len(files)}",
            f"Set expected_files_changed: {len(files)}",
        )


def check_leakage(task: Task, report: Report) -> None:
    """The instruction must not name the hidden tests (5.4 rule 4).

    Two directories are called `tests/` and only one of them is secret, so this check has to
    tell them apart rather than flag the word. See "The two test directories" in
    docs/task_authoring_guide.md: `<task>/tests/` is the hidden grading suite and may never be
    mentioned; `<task>/repo/tests/` is the project's own suite, which the agent can read and
    which a `test_update` task exists to make it edit. A mention that resolves to a real file
    under `repo/` is therefore normal prose, not a leak.
    """
    instruction = str(task.data.get("instruction", ""))
    lowered = instruction.lower()

    for rel in task.test_files:
        name = Path(rel).name
        if name.lower() in lowered or rel.lower() in lowered:
            report.error(
                "LEAK_TESTNAME",
                task.id,
                f"instruction names the hidden test file '{name}'",
                "Describe the symptom, not the test. The agent must never learn the hidden suite's shape.",
            )

    for mention in _TESTS_PATH_MENTION.findall(instruction):
        if _resolves_inside_the_repo(task, mention):
            continue
        report.warn(
            "LEAK_TESTDIR",
            task.id,
            f"instruction mentions '{mention}', which is not a path inside repo/",
            "If this means the hidden suite, remove it. If it means a file the agent can see, "
            "write the path exactly as the agent would (relative to the repo root).",
        )


#: Path-shaped tokens in prose that name a `tests/` or `test/` directory.
_TESTS_PATH_MENTION = re.compile(r"[A-Za-z0-9_.\-]*\btests?[/\\][A-Za-z0-9_./\\-]*")


def _resolves_inside_the_repo(task: Task, mention: str) -> bool:
    """True if `mention` names something that really exists under `repo/`."""
    rel = normalise(mention).strip("/").removeprefix("repo/").rstrip(".,;:")
    return bool(rel) and (task.repo_dir / rel).exists()


def check_smoke(task: Task, report: Report) -> None:
    """Visible smoke tests must not be copies of the hidden suite (5.5: run_tests must not leak the grade)."""
    if not task.smoke_dir.is_dir():
        return
    hidden = {
        (task.path / rel).read_text(encoding="utf-8", errors="replace")
        for rel in task.test_files
        if (task.path / rel).is_file()
    }
    for smoke_file in sorted(task.smoke_dir.rglob("*.py")):
        if smoke_file.read_text(encoding="utf-8", errors="replace") in hidden:
            report.error(
                "SMOKE_LEAK",
                task.id,
                f"smoke/{smoke_file.name} is byte-identical to a hidden test",
                "run_tests() would hand the agent the grading signal. Weaken the smoke test.",
            )


def check_provenance(task: Task, report: Report) -> None:
    prov = task.data.get("provenance", {})
    licence = prov.get("license", "")
    if licence and licence not in PERMISSIVE_LICENCES:
        report.warn(
            "LICENCE",
            task.id,
            f"licence '{licence}' is not on the permissive allow-list",
            f"Confirm it permits redistribution. Known-good: {', '.join(sorted(PERMISSIVE_LICENCES))}",
        )
    if prov.get("source") == "real_repo":
        origin = prov.get("origin", "")
        if " @ " not in origin:
            report.error(
                "ORIGIN_FORMAT",
                task.id,
                f"provenance.origin '{origin}' has no ' @ <commit>' part",
                "Format: 'github.com/<org>/<repo> @ <commit-or-tag>'. Without a commit it is not reproducible.",
            )


def blocking_errors(report: Report, task_id: str) -> list[Finding]:
    """Findings that must stop the behaviour check from running.

    Only errors block. A warning must never suppress it: LOC_BIG is advisory, and gating on
    every finding meant an oversized repo silently skipped its own solvable/failable checks
    and its floor measurement - the validator reported `ok` for a task it had not run.
    """
    return [f for f in report.for_task(task_id) if f.level == "error"]


def check_behaviour(task: Task, report: Report, *, fix: bool = False) -> None:
    """The two checks that actually matter: solvable by gold, failing without it (5.4 rules 1-2)."""
    tid = task.id
    try:
        without = grade(task, apply_gold=False)
    except (TaskError, PatchError) as e:
        report.error("RUN_BASE", tid, f"could not run the hidden tests on the untouched repo: {e}")
        return

    if without.collection_error:
        report.error(
            "COLLECT",
            tid,
            f"pytest collected nothing from {task.test_files}",
            "Usually an import error in the test file. Reproduce with:\n"
            f"         uv run python scripts/verify_gold_patches.py --task {tid} --keep\n"
            f"         ...then read the tail:\n{_indent(without.stdout_tail)}",
        )
        return

    if without.collected != task.n_tests:
        if fix and rewrite_scalar(task, r"^(\s*n_tests:\s*)\d+\s*$", without.collected):
            report.note(f"{tid}: verification.n_tests {task.n_tests} -> {without.collected}")
        else:
            report.error(
                "N_TESTS",
                tid,
                f"verification.n_tests is {task.n_tests}, pytest collects {without.collected}",
                f"Set n_tests: {without.collected}",
            )

    # The reward floor, as a count. Measurement only - reward_v1 is untouched, and ADR 0006's
    # normalisation is still Proposed. Recording it now means the number that a future
    # normalisation would divide by is a checked measurement rather than something an author
    # typed, and it makes "how much headroom does this task have" answerable from task.yaml.
    declared_floor = task.data.get("verification", {}).get("floor_tests")
    if declared_floor is not None and declared_floor != without.passed:
        if fix and rewrite_scalar(task, r"^(\s*floor_tests:\s*)\d+\s*$", without.passed):
            report.note(f"{tid}: verification.floor_tests {declared_floor} -> {without.passed}")
        else:
            report.error(
                "FLOOR",
                tid,
                f"verification.floor_tests is {declared_floor}, the untouched repo passes {without.passed}",
                f"Set floor_tests: {without.passed}",
            )

    if without.all_passed:
        report.error(
            "NOT_FAILABLE",
            tid,
            "the untouched repo passes every hidden test",
            "The bug is not actually reachable by the tests, so reward is 1.0 before the agent does "
            "anything. Strengthen the tests or deepen the bug.",
        )

    try:
        with_gold = grade(task, apply_gold=True)
    except (TaskError, PatchError) as e:
        report.error(
            "GOLD_APPLY",
            tid,
            f"gold patch failed to apply: {e}",
            "Regenerate it: uv run python benchmark/tools/make_gold_patch.py --task " + tid,
        )
        return

    if not with_gold.all_passed:
        report.error(
            "NOT_SOLVABLE",
            tid,
            f"with the gold patch applied, only {with_gold.summary()}",
            "The reference solution must score exactly 1.0, or the task's ceiling is below 1 and every "
            "reported number is wrong. Failing tail:\n" + _indent(with_gold.stdout_tail),
        )


def _indent(text: str, width: int = 9) -> str:
    pad = " " * width
    return "\n".join(pad + line for line in (text or "").strip().splitlines()[-12:])


# ------------------------------------------------------------------ corpus checks


def check_corpus(tasks: list[Task], report: Report, *, require_mix: bool) -> None:
    level = report.error if require_mix else report.warn

    ids = Counter(t.id for t in tasks)
    for tid, count in ids.items():
        if count > 1:
            report.error("DUP_ID", tid, f"id used by {count} tasks", "ids must be globally unique across splits.")

    for split in SPLITS:
        in_split = [t for t in tasks if t.split == split]
        if not in_split:
            continue
        counts = Counter(t.category for t in in_split)
        for category, want in REQUIRED_MIX[split].items():
            have = counts.get(category, 0)
            if have != want:
                level(
                    "MIX",
                    f"<{split}>",
                    f"{category}: {have}/{want}",
                    f"AGENT_BUILD_SPEC 5.2 wants {want} {category} tasks in {split}.",
                )

        total = len(in_split)
        diffs = Counter(t.difficulty for t in in_split)
        for difficulty, target in TARGET_DIFFICULTY.items():
            frac = diffs.get(difficulty, 0) / total
            if abs(frac - target) > DIFFICULTY_TOLERANCE:
                level(
                    "DIFFICULTY",
                    f"<{split}>",
                    f"{difficulty}: {frac:.0%} of {split} (target {target:.0%})",
                    "5.2 wants roughly 40/40/20 easy/medium/hard.",
                )

    real = sum(1 for t in tasks if t.data.get("provenance", {}).get("source") == "real_repo")
    frac = real / len(tasks) if tasks else 0.0
    if frac < MIN_REAL_REPO_FRACTION:
        level(
            "PROVENANCE",
            "<corpus>",
            f"only {real}/{len(tasks)} ({frac:.0%}) tasks are source: real_repo",
            f"5.3 targets >={MIN_REAL_REPO_FRACTION:.0%}. Invented bugs skew toward textbook errors.",
        )


# ------------------------------------------------------------------ entry point


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Validate every task (default).")
    group.add_argument("--split", choices=SPLITS, help="Validate one split.")
    group.add_argument("--task", help="Validate a single task by id.")
    parser.add_argument("--no-run", action="store_true", help="Skip the pytest checks (schema and structure only).")
    parser.add_argument(
        "--require-mix", action="store_true", help="Treat corpus mix/provenance shortfalls as errors (the B2 gate)."
    )
    parser.add_argument("--quiet", action="store_true", help="Only print failures.")
    parser.add_argument(
        "--fix-counts",
        action="store_true",
        help="Write back the two purely measured fields (repo.loc, verification.n_tests) "
        "instead of reporting them as errors. Everything else is a judgement and is never touched.",
    )
    args = parser.parse_args()

    validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))
    report = Report()

    if args.task:
        matches = [p for split in SPLITS for p in [TASKS_ROOT / split / args.task] if p.is_dir()]
        if not matches:
            print(f"No task directory named '{args.task}' under {TASKS_ROOT}", file=sys.stderr)
            return 2
        tasks = [load_task(matches[0])]
    else:
        splits = (args.split,) if args.split else SPLITS
        tasks = list(iter_tasks(splits))

    if not tasks:
        print("No tasks found. Create one: uv run python benchmark/tools/new_task.py --help")
        return 0

    print(f"Validating {len(tasks)} task(s) from {TASKS_ROOT.relative_to(REPO_ROOT)}\n")

    for task in tasks:
        schema_ok = check_schema(task, report, validator)
        check_layout(task, report)
        check_loc(task, report, fix=args.fix_counts)
        check_constraints(task, report)
        check_gold_patch(task, report)
        check_leakage(task, report)
        check_smoke(task, report)
        check_provenance(task, report)
        if not args.no_run and schema_ok and not blocking_errors(report, task.id):
            check_behaviour(task, report, fix=args.fix_counts)

        findings = report.for_task(task.id)
        if findings:
            print(f"{task.split}/{task.id}  [{task.category} / {task.difficulty}]")
            for finding in findings:
                print(finding.render())
            print()
        elif not args.quiet:
            print(f"  ok  {task.split}/{task.id}  [{task.category} / {task.difficulty}]")

    check_corpus(tasks, report, require_mix=args.require_mix)
    corpus = [f for f in report.findings if f.task.startswith("<")]
    if corpus:
        print("\nCorpus-level:")
        for finding in corpus:
            print(finding.render())

    n_err, n_warn = len(report.errors), len(report.warnings)
    if report.notes:
        print("\nRewritten by --fix-counts:")
        for note in report.notes:
            print(f"  {note}")
    print(f"\n{'-' * 60}")
    if n_err:
        print(f"FAILED: {n_err} error(s), {n_warn} warning(s) across {len(tasks)} task(s)")
        return 1
    print(f"PASSED: {len(tasks)} task(s), 0 errors, {n_warn} warning(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
