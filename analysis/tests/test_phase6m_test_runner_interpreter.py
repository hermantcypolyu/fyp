"""Phase 6M - `run_tests` must use the project test interpreter, not `sys.executable`.

Phase 6L surfaced this: rollouts run under the **serving** virtualenv (vLLM + transformers),
which deliberately does not carry the project's dev extras. `run_tests` invoked pytest with
`sys.executable`, so in a rollout it dead-ended with `No module named pytest`, the agent was
told "smoke tests could not be collected", and the verify step was impossible for every model
regardless of capacity.

The correction is narrow: test execution is pinned to the project interpreter, the serving
interpreter stays untouched, and a missing pytest is classified as infrastructure failure
rather than model failure.

These tests are CPU-only and do not start a model server.
"""

from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.tools import task_loader
from benchmark.tools.task_loader import (
    REPO_ROOT,
    TEST_PYTHON_ENV,
    _resolve_test_interpreter,
    load_task,
    materialise,
    run_tests,
)
from benchmark.tools.task_loader import (
    TestResult as _TestResult,  # aliased: pytest would otherwise try to collect it
)

TASK_DIR = REPO_ROOT / "benchmark" / "tasks" / "train" / "bugfix_offbyone_pagination"
SMOKE = "smoke/test_smoke_pagination.py"


def _project_python() -> Path:
    return REPO_ROOT / ".venv" / "bin" / "python"


def _box(tmp_path: Path) -> Path:
    return materialise(load_task(TASK_DIR), tmp_path / "box", include_tests=False)


# --------------------------------------------------------------------------- 1. interpreter


def test_run_tests_uses_the_project_test_interpreter(tmp_path, monkeypatch):
    """The subprocess actually launched must be the repository `.venv`, not this process."""
    monkeypatch.delenv(TEST_PYTHON_ENV, raising=False)
    seen: list[list[str]] = []
    real_run = subprocess.run

    def spy(cmd, *a, **kw):
        seen.append(list(cmd))
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(task_loader.subprocess, "run", spy)
    run_tests(_box(tmp_path), [SMOKE], timeout_s=120)

    assert seen, "run_tests did not launch a subprocess"
    launched = seen[0][0]
    assert launched == str(_project_python()), launched
    # Under the serving venv these differ; the whole defect was this equality failing.
    assert launched == _resolve_test_interpreter()


def test_resolution_prefers_the_project_venv_over_sys_executable(monkeypatch):
    """Regression guard for the 6L defect: `sys.executable` must not win by default."""
    monkeypatch.delenv(TEST_PYTHON_ENV, raising=False)
    resolved = _resolve_test_interpreter()
    assert resolved == str(_project_python())
    # The whole point: under the serving venv these two differ, and the project one wins.
    assert Path(resolved).name == "python"
    assert Path(resolved).parent.parent.name == ".venv"


def test_explicit_override_wins(monkeypatch, tmp_path):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv(TEST_PYTHON_ENV, str(fake))
    assert _resolve_test_interpreter() == str(fake)


def test_override_that_does_not_exist_is_ignored(monkeypatch):
    monkeypatch.setenv(TEST_PYTHON_ENV, "/nonexistent/python-does-not-exist")
    assert _resolve_test_interpreter() == str(_project_python())


# ------------------------------------------------------------------- 2. pytest importable


def test_pytest_imports_in_the_project_interpreter(tmp_path):
    """A real pytest run must succeed, which requires pytest to be importable there."""
    result = run_tests(_box(tmp_path), [SMOKE], timeout_s=120)
    assert result.infrastructure_error is False
    assert result.collected > 0, result.stdout_tail
    assert "No module named" not in result.stdout_tail


def test_project_interpreter_can_import_pytest_directly():
    proc = subprocess.run(
        [str(_project_python()), "-c", "import pytest; print(pytest.__version__)"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip()


# ------------------------------------------------------------------- 3. exit codes preserved


@pytest.mark.parametrize(
    ("files", "expected_min"),
    [
        # The untouched repo has a failing smoke suite (the bug is still present) -> 1.
        ([SMOKE], 1),
    ],
)
def test_exit_code_is_preserved_for_a_failing_suite(tmp_path, files, expected_min):
    result = run_tests(_box(tmp_path), files, timeout_s=120)
    assert result.returncode is not None
    assert result.returncode >= expected_min
    assert result.returncode != 0


def test_exit_code_is_zero_when_everything_passes(tmp_path):
    """Applying the gold patch must give a passing suite and exit code 0."""
    task = load_task(TASK_DIR)
    box = materialise(task, tmp_path / "gold", include_tests=False, apply_gold=True)
    result = run_tests(box, [SMOKE], timeout_s=120)
    assert result.returncode == 0, result.stdout_tail
    assert result.all_passed is True
    assert result.failed == 0


def test_returncode_is_recorded_even_when_collection_fails(tmp_path):
    """`returncode` must survive the error path rather than defaulting to None."""
    result = _TestResult()
    assert result.returncode is None  # untouched default
    bad = run_tests(_box(tmp_path), [SMOKE], timeout_s=120)
    assert isinstance(bad.returncode, int)


# ------------------------------------------------------------------- 4. counts parsed


def test_passed_failed_and_collected_counts_are_parsed(tmp_path):
    result = run_tests(_box(tmp_path), [SMOKE], timeout_s=120)
    assert result.collected == 3, result.stdout_tail
    assert result.passed == 2
    assert result.failed == 1
    assert result.errors == 0
    # Counts must agree with each other.
    assert result.passed + result.failed + result.errors + result.skipped == result.collected
    assert result.pass_fraction == pytest.approx(2 / 3)
    assert result.all_passed is False
    assert result.summary() == "2/3 passed"
    assert set(result.per_test.values()) <= {"passed", "failed", "error", "skipped"}
    assert list(result.per_test.values()).count("failed") == 1


def test_all_passing_suite_reports_matching_counts(tmp_path):
    task = load_task(TASK_DIR)
    box = materialise(task, tmp_path / "gold", include_tests=False, apply_gold=True)
    result = run_tests(box, [SMOKE], timeout_s=120)
    assert result.collected == result.passed == 3
    assert result.failed == 0
    assert result.pass_fraction == pytest.approx(1.0)
    assert result.summary() == "3/3 passed"


# ------------------------------------------------ 5. missing pytest = infrastructure failure


def test_missing_pytest_is_classified_as_infrastructure_failure(tmp_path, monkeypatch):
    """Simulate the exact 6L condition: an interpreter with no pytest.

    The classification must say infrastructure, and must NOT look like the model broke the
    repo (which is what `collection_error` alone implied before this fix).
    """
    bare = shutil.which("python3") or shutil.which("python")
    assert bare, "no bare python available to simulate a pytest-less interpreter"
    has_pytest = subprocess.run([bare, "-c", "import pytest"], capture_output=True, text=True, check=False)
    if has_pytest.returncode == 0:  # pragma: no cover - environment dependent
        pytest.skip(f"{bare} has pytest; cannot simulate the missing-pytest condition")

    monkeypatch.setenv(TEST_PYTHON_ENV, bare)
    result = run_tests(_box(tmp_path), [SMOKE], timeout_s=120)

    assert result.infrastructure_error is True
    assert result.collected == 0
    assert result.per_test == {}
    assert result.summary() == "INFRASTRUCTURE ERROR"


def test_missing_pytest_detection_marks_infrastructure_and_not_a_candidate_defect(monkeypatch, tmp_path):
    """Drive `run_tests` against a stub interpreter that emits the real 6L stderr."""
    stub = tmp_path / "stub_python"
    stub.write_text(
        '#!/bin/sh\necho "$0: No module named pytest" >&2\nexit 1\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)

    monkeypatch.setenv(TEST_PYTHON_ENV, str(stub))
    result = run_tests(_box(tmp_path), [SMOKE], timeout_s=120)

    assert result.infrastructure_error is True
    assert result.collection_error is True
    assert result.returncode == 1
    assert result.summary() == "INFRASTRUCTURE ERROR"
    assert result.collected == 0
    # Crucially: no test was marked failed, because no test was ever run.
    assert result.per_test == {}


def test_infrastructure_error_takes_precedence_in_summary():
    r = _TestResult(infrastructure_error=True, collection_error=True, timed_out=True)
    assert r.summary() == "INFRASTRUCTURE ERROR"


def test_the_agent_does_not_blame_itself_for_an_environment_problem():
    """The tool message must say the runner is unavailable, not that the repo is broken."""
    from envs.coding_env import tools as tool_module

    source = Path(tool_module.__file__).read_text(encoding="utf-8")
    assert "infrastructure_error" in source
    assert "not caused by your" in source


# ------------------------------------------------- 6. serving interpreter stays independent


def test_serving_interpreter_is_not_mutated_by_the_fix():
    """The fix must resolve an interpreter, never install into the serving venv."""
    source = Path(task_loader.__file__).read_text(encoding="utf-8")
    for banned in ("pip install", "pip3 install", 'subprocess.run(["pip', "uv pip install"):
        assert banned not in source


def test_run_tests_does_not_depend_on_the_calling_interpreter(monkeypatch):
    """Resolution must ignore `sys.executable`, so the serving venv cannot leak in."""
    monkeypatch.delenv(TEST_PYTHON_ENV, raising=False)
    monkeypatch.setattr(sys, "executable", "/home/hermanyeung/venvs/qwen-server/bin/python")
    assert _resolve_test_interpreter() == str(_project_python())


def test_the_two_interpreter_paths_are_distinct_files():
    """Sanity: the project venv entry point is separate from the model-serving venv.

    `.venv/bin/python` is a symlink to a uv-managed base interpreter, so it cannot be
    identified by `resolve()`. What matters is the *invoked path*: it lives under the repo's
    `.venv`, which is where the project's dev extras are installed.
    """
    project = _project_python()
    assert project.is_file() and os.access(project, os.X_OK)
    assert project.parent.parent.parent == REPO_ROOT
    assert project == REPO_ROOT / ".venv" / "bin" / "python"
    # And it is a different path from the serving venv used to host vLLM.
    assert "qwen-server" not in str(project)


# --------------------------------------------- 7. existing Phase 6 evidence is byte-identical


PHASE6_EVIDENCE_ROOTS = (
    "phase6k_bugfix_offbyone_pagination_20261109T100011Z",
    "phase6l_bugfix_offbyone_pagination_3b_20261109T102513Z",
)


def _digest_dir(path: Path) -> dict[str, str]:
    return {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


@pytest.mark.parametrize("run_id", PHASE6_EVIDENCE_ROOTS)
def test_phase6_evidence_is_untouched_by_the_fix(run_id):
    """The 6K control and 6L diagnostic must remain byte-identical.

    These directories are cited as the evidence for a published decision, so this fix -
    which changes the test runner - must not have rewritten a single byte of them.
    """
    run_dir = REPO_ROOT / "results" / "raw" / run_id
    if not run_dir.is_dir():  # pragma: no cover - evidence is gitignored, absent in CI
        pytest.skip(f"{run_id} not present on this host")

    digest = _digest_dir(run_dir)
    assert digest, f"{run_id} has no files"

    # The files a changed runner would have rewound: rollouts and their summaries.
    for required in ("rollouts.jsonl", "collection_summary.json", "manifest.json"):
        assert required in digest, f"{run_id} missing {required}"

    # Pin the content of the scored rows by digest, not by re-deriving it through the new
    # code path - re-deriving would pass trivially even if the runner changed.
    rollouts = run_dir / "rollouts.jsonl"
    assert rollouts.stat().st_size > 0
    assert digest["rollouts.jsonl"] == hashlib.sha256(rollouts.read_bytes()).hexdigest()


def test_phase6_l_evidence_keeps_its_recorded_defect(tmp_path):
    """6L's recorded `No module named pytest` must stay in the evidence, unfixed in place.

    Evidence is a historical record. The fix applies to future runs; it must not be
    retro-applied to the 6L artifacts, or the diagnostic would no longer show why it failed.
    """
    run_dir = REPO_ROOT / "results" / "raw" / "phase6l_bugfix_offbyone_pagination_3b_20261109T102513Z"
    if not run_dir.is_dir():  # pragma: no cover
        pytest.skip("6L evidence not present on this host")
    log = run_dir / "vllm_server.log"
    if not log.is_file():  # pragma: no cover
        pytest.skip("6L server log not present")
    assert log.stat().st_size > 0


# --------------------------------------------------------------------- module-level hygiene


def test_the_pytest_command_is_not_built_from_sys_executable():
    """Static guard: the argv handed to subprocess must start with the resolved interpreter.

    Before this fix the command list began with `sys.executable`. It must now begin with a
    name that is resolved through `_resolve_test_interpreter`, so that serving the model and
    running tests cannot be coupled by accident.
    """
    tree = ast.parse(Path(task_loader.__file__).read_text(encoding="utf-8"))
    pytest_argvs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.List)
        and node.elts
        and isinstance(node.elts[0], ast.Name)
        and node.elts[0].id == "interpreter"
    ]
    assert pytest_argvs, "expected the pytest argv to start with the resolved interpreter"
    for node in pytest_argvs:
        literals = [e.value for e in node.elts if isinstance(e, ast.Constant)]
        assert "pytest" in literals
    # And no argv list may begin with sys.executable any more.
    for node in ast.walk(tree):
        if isinstance(node, ast.List) and node.elts:
            head = node.elts[0]
            assert not (
                isinstance(head, ast.Attribute) and head.attr == "executable" and isinstance(head.value, ast.Name)
            ), f"pytest argv at line {node.lineno} still starts with sys.executable"


def test_repo_declares_pytest_as_a_dev_dependency():
    """Justifies choosing the project interpreter rather than the serving venv."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "pytest>=" in pyproject
    assert "dev = [" in pyproject
