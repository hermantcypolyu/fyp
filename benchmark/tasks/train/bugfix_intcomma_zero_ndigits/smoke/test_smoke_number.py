"""Visible smoke checks. Shallow by design - the module imports and separates thousands.
They say nothing about how the digit-count argument is interpreted.
"""

import humanize


def test_module_imports_and_separates_thousands():
    assert humanize.intcomma(1000000) == "1,000,000"


def test_ordinal_still_works():
    assert humanize.ordinal(3) == "3rd"
