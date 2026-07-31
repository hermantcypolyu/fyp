"""Visible smoke checks. Shallow by design - they confirm the module imports and renders
an ordinary whole number of bytes, and say nothing about fractional inputs.
"""

import humanize


def test_import_and_basic_rendering():
    assert humanize.naturalsize(3000000) == "3.0 MB"


def test_binary_mode_still_works():
    assert humanize.naturalsize(3000, True) == "2.9 KiB"
