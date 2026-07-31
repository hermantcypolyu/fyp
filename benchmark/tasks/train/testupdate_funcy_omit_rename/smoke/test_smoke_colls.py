"""Visible smoke checks. Shallow by design - they confirm the library imports and works,
and say nothing about whether the project's own suite has been brought up to date.
"""

from funcy.colls import omit, project


def test_library_imports_and_omits():
    assert omit({"a": 1, "b": 2, "c": 3}, "ac") == {"b": 2}


def test_project_is_the_complement():
    assert project({"a": 1, "b": 2, "c": 3}, "ac") == {"a": 1, "c": 3}
