"""Visible smoke checks. Shallow by design - they confirm the library imports and reverses,
and say nothing about whether the project's own suite has been brought up to date.
"""

import more_itertools as mi


def test_library_imports_and_reverses():
    assert list(mi.always_reversible([1, 2, 3])) == [3, 2, 1]


def test_sibling_helper_still_works():
    assert mi.one(["only"]) == "only"
