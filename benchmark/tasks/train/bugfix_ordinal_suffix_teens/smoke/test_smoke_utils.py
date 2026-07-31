"""Visible smoke checks - the module imports and produces ordinals for small numbers.
Says nothing about behaviour above 100.
"""

from tenacity import _utils


def test_module_imports_and_produces_a_suffix():
    assert _utils.find_ordinal(1) == "st"


def test_to_ordinal_renders():
    assert _utils.to_ordinal(3) == "3rd"
