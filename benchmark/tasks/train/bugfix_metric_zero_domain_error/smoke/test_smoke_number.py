"""Visible smoke checks. Shallow by design - they confirm the module imports and renders
an ordinary reading, and say nothing about which inputs the function rejects.
"""

import humanize


def test_import_and_basic_rendering():
    assert humanize.metric(1500, "V") == "1.50 kV"


def test_scientific_notation_helper_is_available():
    assert humanize.scientific(1000) == "1.00 x 10\u00b3"
