"""Unit tests for the instruction-leakage rules.

Two directories in every task are called `tests/`, and only one of them is secret. The
hidden grading suite at `<task>/tests/` decides the reward and the agent must never learn
its shape; the project's own suite at `<task>/repo/tests/` is ordinary source the agent can
read, and a `test_update` task exists precisely to make it edit that one.

An earlier version of `check_leakage` warned on the *word* `tests/`, which meant every
honest `test_update` instruction carried a permanent false positive - the fastest way to
teach an author to ignore warnings. These tests pin the distinction down so the rule can
stay silent when it should and still fire when it matters.
"""

from __future__ import annotations

import pytest

from benchmark.schema.validate_tasks import Report, check_leakage
from benchmark.tools.task_loader import Task


def make_task(tmp_path, instruction: str, *, repo_files=(), hidden="tests/test_hidden.py") -> Task:
    for rel in repo_files:
        path = tmp_path / "repo" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    hidden_path = tmp_path / hidden
    hidden_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_path.write_text("", encoding="utf-8")
    return Task(
        path=tmp_path,
        data={"id": "probe", "instruction": instruction, "verification": {"test_files": [hidden]}},
    )


def codes(task: Task) -> list[str]:
    report = Report()
    check_leakage(task, report)
    return [f.code for f in report.findings]


def test_silent_on_an_ordinary_instruction(tmp_path):
    task = make_task(tmp_path, "Paging past the end of a collection drops the final partial page.")
    assert codes(task) == []


def test_naming_the_hidden_file_is_an_error(tmp_path):
    task = make_task(tmp_path, "Make test_hidden.py pass.")
    assert codes(task) == ["LEAK_TESTNAME"]


def test_naming_the_hidden_path_is_an_error(tmp_path):
    task = make_task(tmp_path, "The suite at tests/test_hidden.py is failing.")
    # Both rules fire: the filename is named, and the path is not inside repo/.
    assert "LEAK_TESTNAME" in codes(task)


def test_a_bare_tests_directory_still_warns(tmp_path):
    task = make_task(tmp_path, "Something under tests/ is failing.")
    assert codes(task) == ["LEAK_TESTDIR"]


def test_a_visible_repo_suite_is_not_a_leak(tmp_path):
    """The whole point of the test_update category."""
    task = make_task(
        tmp_path,
        "The repo's own suite at tests/test_money.py still calls the 1.x API and no longer imports.",
        repo_files=["tests/test_money.py"],
    )
    assert codes(task) == []


def test_a_visible_repo_directory_is_not_a_leak(tmp_path):
    task = make_task(tmp_path, "Bring everything under tests/ up to the new API.", repo_files=["tests/test_a.py"])
    assert codes(task) == []


def test_an_explicit_repo_prefix_is_understood(tmp_path):
    task = make_task(tmp_path, "Update repo/tests/test_money.py.", repo_files=["tests/test_money.py"])
    assert codes(task) == []


def test_a_repo_file_that_does_not_exist_still_warns(tmp_path):
    """A path the agent cannot actually open is either a typo or the hidden suite."""
    task = make_task(tmp_path, "Update tests/test_absent.py.", repo_files=["tests/test_money.py"])
    assert codes(task) == ["LEAK_TESTDIR"]


@pytest.mark.parametrize("mention", ["tests/test_money.py.", "tests/test_money.py,", "tests/test_money.py;"])
def test_trailing_punctuation_does_not_break_resolution(tmp_path, mention):
    task = make_task(tmp_path, f"See {mention}", repo_files=["tests/test_money.py"])
    assert codes(task) == []


def test_singular_test_directory_is_also_recognised(tmp_path):
    task = make_task(tmp_path, "Look in test/helpers.py.", repo_files=["test/helpers.py"])
    assert codes(task) == []
