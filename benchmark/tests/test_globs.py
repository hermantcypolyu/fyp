"""Unit tests for the glob matcher.

This is small but load-bearing: `forbidden_write_globs` is what stands between the agent
and the hidden test suite, and B4 turns a match into a hard-zero reward gate. A matcher
that is one character too permissive silently hands out free rewards; one that is too
strict blocks legitimate edits and makes every task look unsolvable. The subtlety worth
testing is the difference between the hidden suite at `tests/**` and a repo's own visible
suite at `repo/tests/**`, which `test_update` tasks depend on.
"""

from __future__ import annotations

import pytest

from benchmark.tools.globs import normalise, path_matches, path_matches_any

FORBIDDEN = ["tests/**", "solution/**", "**/conftest.py", "task.yaml"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("repo\\src\\a.py", "repo/src/a.py"),
        ("./repo/src/a.py", "repo/src/a.py"),
        ("repo//src///a.py", "repo/src/a.py"),
        ("repo/src/", "repo/src"),
    ],
)
def test_normalise(raw, expected):
    assert normalise(raw) == expected


@pytest.mark.parametrize(
    "path,pattern",
    [
        ("repo/src/a.py", "repo/src/**"),
        ("repo/src/deep/nested/a.py", "repo/src/**"),
        ("repo/a.py", "repo/*.py"),
        ("tests/test_x.py", "tests/**"),
        ("tests/sub/test_x.py", "tests/**"),
        ("repo/conftest.py", "**/conftest.py"),
        ("conftest.py", "**/conftest.py"),
        ("repo/deep/conftest.py", "**/conftest.py"),
        ("task.yaml", "task.yaml"),
        ("repo/pyproject.toml", "repo/pyproject.toml"),
        ("repo/a1.py", "repo/a?.py"),
    ],
)
def test_matches(path, pattern):
    assert path_matches(path, pattern)


@pytest.mark.parametrize(
    "path,pattern",
    [
        # `*` must not cross a directory boundary
        ("repo/src/deep/a.py", "repo/src/*.py"),
        ("repo/src/a.py", "repo/*.py"),
        # a prefix is not a match
        ("repository/src/a.py", "repo/**"),
        ("tests_helper/x.py", "tests/**"),
        # the pattern is anchored at both ends
        ("a/task.yaml", "task.yaml"),
        ("repo/conftest.pyc", "**/conftest.py"),
    ],
)
def test_does_not_match(path, pattern):
    assert not path_matches(path, pattern)


def test_hidden_suite_and_a_repos_own_suite_are_different_paths():
    """The crux of the `test_update` category. `tests/**` names the hidden grading suite,
    which no task may ever write to. `repo/tests/**` is the project's own visible suite,
    which is exactly what a test_update task asks the agent to edit. If these two ever
    collided, every test_update task would score a hard zero for doing the right thing.
    """
    assert path_matches_any("tests/test_money_suite.py", FORBIDDEN)
    assert not path_matches_any("repo/tests/test_money.py", FORBIDDEN)
    assert path_matches("repo/tests/test_money.py", "repo/tests/**")


def test_forbidden_set_catches_every_route_to_the_grader():
    for path in (
        "tests/test_x.py",
        "tests/helpers/data.json",
        "solution/gold.patch",
        "solution/fixed/src/a.py",
        "task.yaml",
        "repo/conftest.py",
    ):
        assert path_matches_any(path, FORBIDDEN), path


def test_forbidden_set_allows_ordinary_work():
    for path in (
        "repo/src/a.py",
        "repo/README.md",
        "repo/pyproject.toml",
        "repo/tests/test_money.py",
        "smoke/test_smoke.py",
    ):
        assert not path_matches_any(path, FORBIDDEN), path


def test_backslashes_are_matched_the_same_as_forward_slashes():
    """The agent may well emit Windows separators; the gate must not be bypassable by
    spelling the path differently."""
    assert path_matches_any("tests\\test_x.py", FORBIDDEN)
    assert path_matches_any(".\\tests\\test_x.py", FORBIDDEN)


def test_empty_pattern_list_matches_nothing():
    assert not path_matches_any("anything.py", [])
