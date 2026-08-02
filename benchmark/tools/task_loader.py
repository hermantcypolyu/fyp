"""Load, materialise and grade benchmark tasks.

This is the one module that knows the on-disk layout of a task. `validate_tasks.py`,
`verify_gold_patches.py`, the smoke test and (in B3) the resources server's verifier all
go through it, so there is exactly one definition of "run the hidden tests and count what
passed".

That single definition is what makes AGENT_BUILD_SPEC 5.6's critical invariant true by
construction: the same code produces the evaluation metric and the RL reward, because
there is only one copy of it.

Task layout (5.1):

    benchmark/tasks/{split}/{task_id}/
        task.yaml
        repo/               what the agent sees and may edit
        tests/              HIDDEN - never mounted into the sandbox
        smoke/              optional, visible, safe for the agent to run
        solution/gold.patch
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from shutil import copytree, rmtree
from typing import Any

import yaml

from benchmark.tools.patch_utils import apply_patch

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_ROOT = REPO_ROOT / "benchmark" / "tasks"
SPLITS = ("train", "heldout")

#: pytest must not inherit this project's own pyproject.toml (testpaths, addopts, ...).
_ISOLATED_INI = "[pytest]\naddopts =\n"

_IGNORE = ("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")


class TaskError(Exception):
    """Something is wrong with a task on disk. The message names the file and the fix."""


# --------------------------------------------------------------------------- loading


@dataclass
class Task:
    path: Path
    data: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.data.get("id", self.path.name))

    @property
    def split(self) -> str:
        return str(self.data.get("split", self.path.parent.name))

    @property
    def category(self) -> str:
        return str(self.data.get("category", "?"))

    @property
    def difficulty(self) -> str:
        return str(self.data.get("difficulty", "?"))

    @property
    def repo_dir(self) -> Path:
        return self.path / "repo"

    @property
    def tests_dir(self) -> Path:
        return self.path / "tests"

    @property
    def smoke_dir(self) -> Path:
        return self.path / "smoke"

    @property
    def gold_patch_path(self) -> Path:
        return self.path / "solution" / "gold.patch"

    @property
    def test_files(self) -> list[str]:
        return list(self.data.get("verification", {}).get("test_files", []))

    @property
    def timeout_s(self) -> int:
        return int(self.data.get("verification", {}).get("timeout_s", 60))

    @property
    def n_tests(self) -> int:
        return int(self.data.get("verification", {}).get("n_tests", 0))

    def gold_patch(self) -> str:
        if not self.gold_patch_path.is_file():
            raise TaskError(f"{self.id}: missing solution/gold.patch")
        return self.gold_patch_path.read_text(encoding="utf-8")


def load_task(task_dir: Path) -> Task:
    task_dir = Path(task_dir)
    yaml_path = task_dir / "task.yaml"
    if not yaml_path.is_file():
        raise TaskError(f"{task_dir}: no task.yaml")
    try:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise TaskError(f"{yaml_path}: not valid YAML\n  {e}") from e
    if not isinstance(data, dict):
        raise TaskError(f"{yaml_path}: top level must be a mapping, got {type(data).__name__}")
    return Task(path=task_dir, data=_stringify_dates(data))


def _stringify_dates(value: Any) -> Any:
    """YAML turns an unquoted `2026-07-31` into a `datetime.date`, which JSON Schema then
    rejects as "not of type string". Rather than making every author remember quotes,
    normalise dates to ISO strings on the way in."""
    if isinstance(value, dict):
        return {k: _stringify_dates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_dates(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    return value


def iter_tasks(splits: tuple[str, ...] = SPLITS, root: Path = TASKS_ROOT) -> Iterator[Task]:
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for task_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            yield load_task(task_dir)


# --------------------------------------------------------------------------- metrics


_BLANK_OR_COMMENT = re.compile(r"^\s*(#.*)?$")


def count_loc(repo_dir: Path) -> int:
    """Non-blank, non-comment lines of Python under repo/. Docstrings count as code."""
    total = 0
    for path in sorted(repo_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not _BLANK_OR_COMMENT.match(line):
                total += 1
    return total


# --------------------------------------------------------------- materialise & grade


def materialise(
    task: Task,
    dest: Path,
    *,
    include_tests: bool = True,
    include_smoke: bool = True,
    apply_gold: bool = False,
) -> Path:
    """Build a disposable working copy of a task at `dest`. Returns `dest`.

    `include_tests=False` is how the agent's sandbox is built: the hidden suite is simply
    never written to disk inside the agent's world, so isolation does not depend on
    permissions being right (B3's hard gate).
    """
    dest = Path(dest)
    if dest.exists():
        rmtree(dest)
    dest.mkdir(parents=True)

    ignore = _ignore_patterns()
    copytree(task.repo_dir, dest / "repo", ignore=ignore)
    if include_smoke and task.smoke_dir.is_dir():
        copytree(task.smoke_dir, dest / "smoke", ignore=ignore)
    if include_tests and task.tests_dir.is_dir():
        copytree(task.tests_dir, dest / "tests", ignore=ignore)
    if apply_gold:
        apply_patch(dest, task.gold_patch())
    return dest


def _ignore_patterns():
    from shutil import ignore_patterns

    return ignore_patterns(*_IGNORE)


@dataclass
class TestResult:
    """Structured outcome of one pytest run. This is the raw material of reward_v1."""

    collected: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    per_test: dict[str, str] = field(default_factory=dict)  # nodeid -> passed|failed|error|skipped
    timed_out: bool = False
    collection_error: bool = False
    returncode: int | None = None
    stdout_tail: str = ""

    @property
    def pass_fraction(self) -> float:
        """Fraction of tests passing. Zero if nothing could even be collected."""
        return self.passed / self.collected if self.collected else 0.0

    @property
    def all_passed(self) -> bool:
        return self.collected > 0 and self.passed == self.collected

    def summary(self) -> str:
        if self.timed_out:
            return "TIMEOUT"
        if self.collection_error:
            return "COLLECTION ERROR"
        return f"{self.passed}/{self.collected} passed"


def run_tests(
    work_dir: Path,
    test_files: list[str],
    *,
    timeout_s: int = 60,
    tail_chars: int = 2000,
    extra_pythonpath: Sequence[Path] = (),
    env_overrides: dict[str, str] | None = None,
    preexec_fn: Callable[[], None] | None = None,
) -> TestResult:
    """Run pytest inside a materialised working copy.

    cwd and PYTHONPATH are both `work_dir/repo`, so tests import the repo the same way a
    user would. Config inheritance is cut off with a generated ini, and the environment is
    scrubbed of the variables that make runs non-deterministic.

    The last three arguments are the hardening hooks used by the graded path
    (`envs.coding_env.verifier`): an extra PYTHONPATH entry carrying a `sitecustomize.py`
    network guard, extra environment variables, and a POSIX `setrlimit` callback. They are
    optional and default to off so that authoring tools keep the cheap path - but they hang
    off *this* function deliberately, because 5.6's critical invariant is that evaluation
    and the RL reward come from the same runner. A second, hardened copy of this code would
    be free to drift, and the study would be invalid the moment it did.
    """
    work_dir = Path(work_dir).resolve()
    repo_dir = work_dir / "repo"
    missing = [f for f in test_files if not (work_dir / f).is_file()]
    if missing:
        raise TaskError(f"{work_dir}: test file(s) not materialised: {', '.join(missing)}")

    ini = work_dir / "_isolated_pytest.ini"
    ini.write_text(_ISOLATED_INI, encoding="utf-8")
    junit = work_dir / "_junit.xml"

    path_entries = [str(Path(p).resolve()) for p in extra_pythonpath] + [str(repo_dir)]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTEST_", "HYPOTHESIS_"))}
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(path_entries),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",  # determinism (5.4 rule 3)
            "PYTHONWARNINGS": "ignore",
            "NO_COLOR": "1",
            "COLUMNS": "100",
        }
    )
    env.update(env_overrides or {})

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(ini),
        "-p",
        "no:cacheprovider",
        "-p",
        "no:randomly",
        "--tb=short",
        "-q",
        "--no-header",
        f"--junit-xml={junit}",
        *[str(work_dir / f) for f in test_files],
    ]

    kwargs: dict[str, Any] = {}
    if preexec_fn is not None:  # POSIX only; ignored on Windows, see envs/coding_env/limits.py
        kwargs["preexec_fn"] = preexec_fn

    result = TestResult()
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd,
            cwd=repo_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            **kwargs,
        )
    except subprocess.TimeoutExpired:
        result.timed_out = True
        result.stdout_tail = f"Test run exceeded the {timeout_s}s timeout."
        return result

    result.returncode = proc.returncode
    output = (proc.stdout or "") + (proc.stderr or "")
    result.stdout_tail = output[-tail_chars:]

    if not junit.is_file():
        # pytest died before it could write a report - almost always a collection error.
        result.collection_error = True
        return result

    _parse_junit(junit, result)
    if result.collected == 0:
        result.collection_error = True
    return result


def _parse_junit(junit: Path, result: TestResult) -> None:
    try:
        root = ET.parse(junit).getroot()  # noqa: S314 - file we just generated ourselves
    except ET.ParseError:
        result.collection_error = True
        return
    for case in root.iter("testcase"):
        nodeid = f"{case.get('classname', '')}::{case.get('name', '')}".lstrip(":")
        if case.find("error") is not None:
            status = "error"
        elif case.find("failure") is not None:
            status = "failed"
        elif case.find("skipped") is not None:
            status = "skipped"
        else:
            status = "passed"
        result.per_test[nodeid] = status

    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for status in result.per_test.values():
        counts[status] += 1
    result.passed, result.failed = counts["passed"], counts["failed"]
    result.errors, result.skipped = counts["error"], counts["skipped"]
    result.collected = len(result.per_test)


def grade(task: Task, *, apply_gold: bool, work_dir: Path | None = None) -> TestResult:
    """Materialise the task, optionally apply the gold patch, run the hidden tests.

    The two calls that matter:
      grade(task, apply_gold=True)   -> must be all_passed   (task is solvable)
      grade(task, apply_gold=False)  -> must NOT be all_passed (task is failable)
    """
    if work_dir is not None:
        work = materialise(task, work_dir, apply_gold=apply_gold)
        return run_tests(work, task.test_files, timeout_s=task.timeout_s)

    tmp = Path(tempfile.mkdtemp(prefix=f"task_{task.id}_"))
    try:
        work = materialise(task, tmp / "work", apply_gold=apply_gold)
        return run_tests(work, task.test_files, timeout_s=task.timeout_s)
    finally:
        rmtree(tmp, ignore_errors=True)
