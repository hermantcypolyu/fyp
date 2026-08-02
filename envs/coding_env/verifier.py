"""The verifier: agent's repo + pristine hidden tests -> structured result -> reward_v1.

⭐ 5.6's critical invariant is that the *same* verifier produces the evaluation metric and
the RL reward. This module is that one implementation; both paths call `verify()`.

**Nothing in the sandbox is trusted for scoring.** The grading copy is built fresh from
`benchmark/tasks/<split>/<id>/`, so the hidden suite that runs is byte-identical to the one
in git, and only then is the agent's `repo/` copied over it. An agent that somehow put a
`tests/` directory in its sandbox has changed nothing about what is executed here.

reward_v1 (5.7) is the fraction of hidden tests passing, zeroed outright - not penalised -
by any hard gate:

| Gate | Trigger |
|---|---|
| `tests_modified` | wrote to a `forbidden_write_globs` path, or planted a pytest hook |
| `no_submit` | the episode ended without `submit` |
| `sandbox_escape` | wrote outside the sandbox root |
| `syntax_broken` | some file under `repo/` no longer parses |
| `timeout` | the graded run hit the wall clock |

A timeout is a *failure*, not an error: reward 0, recorded, and the episode still produces
a row. Treating it as an exception would quietly drop the slowest rollouts out of the
dataset, which biases every aggregate computed from it.
"""

from __future__ import annotations

import ast
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copytree, ignore_patterns, rmtree
from typing import Any

from benchmark.tools.task_loader import Task, TestResult, materialise, run_tests
from envs.coding_env.limits import ResourceLimits, network_guard_dir, no_network_env
from envs.coding_env.sandbox import Sandbox
from envs.coding_env.tools import GATE_SANDBOX_ESCAPE, GATE_TESTS_MODIFIED

__all__ = [
    "GATE_NO_SUBMIT",
    "GATE_SANDBOX_ESCAPE",
    "GATE_SYNTAX_BROKEN",
    "GATE_TESTS_MODIFIED",
    "GATE_TIMEOUT",
    "VerifyResult",
    "verify",
    "verify_sandbox",
]

# `tests_modified` and `sandbox_escape` are raised by the toolbox, so their names are
# imported rather than restated - two spellings of one gate would silently stop zeroing.
GATE_NO_SUBMIT = "no_submit"
GATE_SYNTAX_BROKEN = "syntax_broken"
GATE_TIMEOUT = "timeout"

ALL_GATES = (GATE_TESTS_MODIFIED, GATE_NO_SUBMIT, GATE_SANDBOX_ESCAPE, GATE_SYNTAX_BROKEN, GATE_TIMEOUT)

#: Files that change what pytest itself does. An agent may not introduce one; if a task's
#: own repo ships one, it is kept, since then it is part of the problem, not an attack.
_PYTEST_HOOKS = ("conftest.py", "sitecustomize.py", "pytest.ini", "tox.ini", "setup.cfg")

_IGNORE = ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.egg-info")


@dataclass
class VerifyResult:
    """One graded episode. This is the row that reward, metrics and traces all read."""

    task_id: str
    reward: float
    pass_fraction: float
    passed: int = 0
    collected: int = 0
    per_test: dict[str, str] = field(default_factory=dict)
    gates: list[str] = field(default_factory=list)
    submitted: bool = False
    tool_calls: int = 0
    limits_enforced: dict[str, bool] = field(default_factory=dict)
    stdout_tail: str = ""

    @property
    def zeroed(self) -> bool:
        return bool(self.gates)

    @property
    def timed_out(self) -> bool:
        return GATE_TIMEOUT in self.gates

    def summary(self) -> str:
        base = f"{self.task_id}: {self.passed}/{self.collected} hidden tests, reward {self.reward:.2f}"
        return f"{base} [ZEROED: {', '.join(self.gates)}]" if self.gates else base

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reward": self.reward,
            "pass_fraction": self.pass_fraction,
            "passed": self.passed,
            "collected": self.collected,
            "per_test": dict(self.per_test),
            "gates": list(self.gates),
            "submitted": self.submitted,
            "tool_calls": self.tool_calls,
            "limits_enforced": dict(self.limits_enforced),
        }


def verify_sandbox(sandbox: Sandbox, *, limits: ResourceLimits | None = None, keep: Path | None = None) -> VerifyResult:
    """Grade a finished episode. Reads `submitted` and the raised gates off the toolbox."""
    box = sandbox.toolbox
    return verify(
        sandbox.task,
        sandbox.repo_dir,
        submitted=box.submitted,
        gates=box.gates,
        tool_calls=box.n_calls,
        limits=limits,
        keep=keep,
    )


def verify(
    task: Task,
    agent_repo: Path,
    *,
    submitted: bool,
    gates: Iterable[str] = (),
    tool_calls: int = 0,
    limits: ResourceLimits | None = None,
    keep: Path | None = None,
) -> VerifyResult:
    """Run the pristine hidden suite against `agent_repo` and score it.

    `keep` writes the grading directory somewhere durable instead of a temp dir - used by
    tests and by post-mortems of a surprising reward.
    """
    limits = limits or ResourceLimits(wall_clock_s=task.timeout_s, cpu_s=task.timeout_s)
    raised = set(gates)
    if not submitted:
        raised.add(GATE_NO_SUBMIT)

    base = Path(keep) if keep is not None else Path(tempfile.mkdtemp(prefix=f"verify_{task.id}_"))
    try:
        grade_root = _build_grading_copy(task, agent_repo, base)
        if _has_injected_hook(task, grade_root / "repo"):
            raised.add(GATE_TESTS_MODIFIED)
        if _broken_syntax(grade_root / "repo"):
            raised.add(GATE_SYNTAX_BROKEN)

        guard = network_guard_dir(base)
        result: TestResult = run_tests(
            grade_root,
            task.test_files,
            timeout_s=limits.wall_clock_s,
            extra_pythonpath=[guard],
            env_overrides=no_network_env(),
            preexec_fn=limits.preexec_fn(),
        )
    finally:
        if keep is None:
            rmtree(base, ignore_errors=True)

    if result.timed_out:
        raised.add(GATE_TIMEOUT)

    pass_fraction = result.pass_fraction
    ordered = [g for g in ALL_GATES if g in raised]
    return VerifyResult(
        task_id=task.id,
        reward=0.0 if ordered else pass_fraction,
        pass_fraction=pass_fraction,
        passed=result.passed,
        collected=result.collected,
        per_test=dict(result.per_test),
        gates=ordered,
        submitted=submitted,
        tool_calls=tool_calls,
        limits_enforced=limits.enforced(),
        stdout_tail=result.stdout_tail,
    )


# --------------------------------------------------------------------------- internals


def _build_grading_copy(task: Task, agent_repo: Path, base: Path) -> Path:
    """Pristine tests from git + the agent's repo. Never the sandbox's idea of either."""
    grade_root = base / "grade"
    materialise(task, grade_root, include_tests=True, include_smoke=False)
    rmtree(grade_root / "repo")
    copytree(Path(agent_repo), grade_root / "repo", ignore=_IGNORE)
    return grade_root


def _has_injected_hook(task: Task, repo: Path) -> bool:
    """True if the agent added a file that changes how pytest itself behaves.

    The write globs already forbid `**/conftest.py`, so this should never fire. It is here
    because the cost of missing it is a silently perfect score, and the cost of the check
    is a directory walk.
    """
    pristine = {p.relative_to(task.repo_dir).as_posix() for p in task.repo_dir.rglob("*") if p.is_file()}
    for path in repo.rglob("*"):
        if path.is_file() and path.name in _PYTEST_HOOKS:
            if path.relative_to(repo).as_posix() not in pristine:
                return True
    return False


def _broken_syntax(repo: Path) -> bool:
    """5.7's `syntax_broken`, as "some file under repo/ no longer parses".

    Parsing rather than importing: importing runs module-level code, which for a broken
    repo can hang or write files, and would make the gate depend on import order.
    """
    for path in repo.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            return True
    return False
