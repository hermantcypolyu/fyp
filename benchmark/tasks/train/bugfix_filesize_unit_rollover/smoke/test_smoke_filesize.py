"""Visible smoke checks. Shallow by design - they confirm the module imports and renders
an ordinary value, and say nothing about behaviour at unit boundaries.
"""

import humanize


def test_import_and_basic_rendering():
    assert humanize.naturalsize(3000000) == "3.0 MB"


def test_bytes_below_the_first_suffix():
    assert humanize.naturalsize(300) == "300 Bytes"
