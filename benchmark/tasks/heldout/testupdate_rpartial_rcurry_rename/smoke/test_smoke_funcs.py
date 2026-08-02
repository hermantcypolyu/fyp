"""Visible smoke checks. Shallow by design - they confirm the library imports and works,
and say nothing about whether the project's own suite has been brought up to date.
"""

from operator import __sub__

from funcy.funcs import compose, identity, rpartial


def test_library_imports_and_binds_from_the_right():
    assert rpartial(__sub__, 10)(1) == -9


def test_composition_still_works():
    assert compose(identity)(42) == 42
