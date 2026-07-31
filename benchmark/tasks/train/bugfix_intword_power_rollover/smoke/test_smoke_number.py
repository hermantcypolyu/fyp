"""Visible smoke checks. Shallow by design - they exercise the middle of the range only,
and say nothing about the top of the scale table.
"""

import humanize


def test_module_imports_and_names_a_scale():
    assert humanize.intword("1000000") == "1.0 million"


def test_small_numbers_pass_through():
    assert humanize.intword("100") == "100"
