"""The verifier: reward_v1, the hard-zero gates, and the hardening of the graded run.

The single most important property here is that scoring does **not** read the sandbox's
tests. `test_verifier_ignores_tests_planted_in_the_sandbox` plants a suite that passes
trivially and asserts the real score is unaffected - that is the difference between a
reward signal and a random number generator.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from benchmark.tools.patch_utils import apply_patch
from benchmark.tools.task_loader import TASKS_ROOT, load_task
from envs.coding_env.limits import ResourceLimits, network_guard_dir, no_network_env
from envs.coding_env.sandbox import open_sandbox
from envs.coding_env.tools import GATE_TESTS_MODIFIED
from envs.coding_env.verifier import (
    GATE_NO_SUBMIT,
    GATE_SYNTAX_BROKEN,
    GATE_TEST_COUNT_MISMATCH,
    GATE_TIMEOUT,
    verify,
    verify_sandbox,
)
from rewards.compose import GATE_REWARD

TASK_ID = "bugfix_metric_zero_domain_error"


@pytest.fixture(scope="module")
def task():
    return load_task(TASKS_ROOT / "train" / TASK_ID)


@pytest.fixture
def sb(task, tmp_path):
    with open_sandbox(task, parent=tmp_path) as sandbox:
        yield sandbox


def _solve(sb) -> None:
    apply_patch(sb.root, sb.task.gold_patch())


# --------------------------------------------------------------------------- scoring


@pytest.mark.slow
def test_gold_patch_scores_one(sb):
    """The one end-to-end run: solve the task in the sandbox, submit, grade."""
    _solve(sb)
    sb.toolbox.submit()
    result = verify_sandbox(sb)
    assert result.gates == []
    assert result.reward == 1.0
    assert result.collected == sb.task.n_tests


@pytest.mark.slow
def test_unsolved_task_scores_below_one(sb):
    sb.toolbox.submit()
    result = verify_sandbox(sb)
    assert result.reward < 1.0
    assert result.collected > 0  # a zero here would mean a collection error, not a wrong answer


@pytest.mark.slow
def test_doing_nothing_scores_exactly_zero(sb):
    """ADR 0006: the do-nothing plateau sits at 0.0, not at the task's floor."""
    sb.toolbox.submit()
    result = verify_sandbox(sb)
    assert result.gates == []
    assert result.passed == sb.task.floor_tests
    assert result.reward == 0.0
    assert result.pass_fraction > 0.0  # the raw measurement still shows the floor


@pytest.mark.slow
def test_verifier_ignores_tests_planted_in_the_sandbox(sb):
    """Grading reads the pristine suite from the corpus, never the sandbox's copy."""
    (sb.root / "tests").mkdir()
    (sb.root / "tests" / "test_number.py").write_text("def test_everything():\n    assert True\n", encoding="utf-8")
    sb.toolbox.submit()
    result = verify_sandbox(sb)
    assert result.collected == sb.task.n_tests  # the real 12, not the planted 1
    assert result.reward < 1.0


# --------------------------------------------------------------------------- gates


@pytest.mark.slow
def test_no_submit_zeroes_a_correct_solution(sb):
    """The work was right; the episode was not finished. Recorded, not silently credited."""
    _solve(sb)
    result = verify_sandbox(sb)
    assert result.gates == [GATE_NO_SUBMIT]
    assert result.reward == GATE_REWARD
    assert result.pass_fraction == 1.0  # kept, so gate cost is measurable


@pytest.mark.slow
def test_tests_modified_zeroes_a_correct_solution(sb):
    _solve(sb)
    sb.toolbox.write_file("tests/test_number.py", "assert True")  # refused, but raises the gate
    sb.toolbox.submit()
    result = verify_sandbox(sb)
    assert GATE_TESTS_MODIFIED in result.gates
    assert result.reward == GATE_REWARD


@pytest.mark.slow
def test_syntax_broken_gate(sb):
    sb.toolbox.write_file("repo/humanize/number.py", "def broken(:\n")
    sb.toolbox.submit()
    result = verify_sandbox(sb)
    assert GATE_SYNTAX_BROKEN in result.gates
    assert result.reward == GATE_REWARD


@pytest.mark.slow
def test_injected_pytest_hook_is_caught(sb):
    """Written straight to disk, bypassing the toolbox - the verifier must still catch it."""
    (sb.repo_dir / "conftest.py").write_text("def pytest_collection_modifyitems(items): items.clear()\n", "utf-8")
    sb.toolbox.submit()
    result = verify_sandbox(sb)
    assert GATE_TESTS_MODIFIED in result.gates
    assert result.reward == GATE_REWARD


def test_timeout_is_a_scored_failure_not_an_error(sb):
    """A wall-clock overrun produces a scored row, not an exception."""
    sb.toolbox.submit()
    result = verify(sb.task, sb.repo_dir, submitted=True, limits=ResourceLimits(wall_clock_s=0))
    assert result.timed_out
    assert result.gates == [GATE_TIMEOUT]
    assert result.reward == GATE_REWARD


@pytest.mark.slow
def test_test_count_mismatch_zeroes_even_a_perfect_run(sb):
    """The denominator is pinned to task.yaml, so shrinking it cannot pay.

    Simulated by declaring a count the pristine suite cannot produce - the same signal a
    collection hook, a skip mark or an early `sys.exit` would generate.
    """
    fresh = load_task(TASKS_ROOT / "train" / TASK_ID)
    fresh.data["verification"]["n_tests"] = 999
    _solve(sb)
    result = verify(fresh, sb.repo_dir, submitted=True)
    assert GATE_TEST_COUNT_MISMATCH in result.gates
    assert result.reward == GATE_REWARD
    assert result.pass_fraction == 1.0  # the work was right; the run was still invalid


@pytest.mark.slow
def test_declared_count_matches_reality_for_the_smoke_task(sb):
    """Guards the gate itself: a stale n_tests would zero every episode on that task."""
    sb.toolbox.submit()
    assert verify_sandbox(sb).collected == sb.task.n_tests


# --------------------------------------------------------------------------- limits


def test_limits_are_reported_honestly():
    """Never claim a cap the platform does not apply (spec 1.2)."""
    enforced = ResourceLimits().enforced()
    assert enforced["wall_clock_s"] is True
    assert enforced["network"] is True
    posix = os.name == "posix"
    assert enforced["memory_mb"] is posix
    assert enforced["cpu_s"] is posix


def test_preexec_fn_exists_only_on_posix():
    assert (ResourceLimits().preexec_fn() is not None) is (os.name == "posix")


def test_network_guard_blocks_sockets(tmp_path):
    """Run a real interpreter with the guard on PYTHONPATH and try to open a socket."""
    guard = network_guard_dir(tmp_path)
    env = dict(os.environ)
    env.update(no_network_env())
    env["PYTHONPATH"] = str(guard)
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", "import socket; socket.create_connection(('example.com', 80))"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode != 0
    assert "network access is disabled during grading" in proc.stderr


def test_network_guard_survives_importing_ssl(tmp_path):
    """Regression: replacing socket.socket outright broke `class SSLSocket(socket)`,
    which took out pdb, asyncio and therefore pytest itself."""
    guard = network_guard_dir(tmp_path)
    env = dict(os.environ)
    env.update(no_network_env())
    env["PYTHONPATH"] = str(guard)
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", "import ssl, asyncio, pdb; print('ok')"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "ok" in proc.stdout


def test_network_guard_is_not_active_without_it(tmp_path):
    """Control: the failure above is the guard, not a broken interpreter."""
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", "import socket; print(socket.create_connection)"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0


# --------------------------------------------------------------------------- shape


@pytest.mark.slow
def test_result_serialises_for_the_trace(sb):
    sb.toolbox.submit()
    row = verify_sandbox(sb).to_dict()
    assert set(row) >= {"task_id", "reward", "pass_fraction", "gates", "submitted", "limits_enforced"}
    assert isinstance(row["per_test"], dict)
