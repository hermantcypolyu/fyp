"""Visible smoke checks: the table primitives still behave.

They touch neither the extras table nor the feature gate, so they pass either way.
"""

from reportkit.table import divider, format_row, widths


def test_widths_of_an_empty_table():
    assert widths([]) == []


def test_a_single_row_renders_and_has_a_rule():
    column_widths = widths([["ab", "c"]])
    assert format_row(["ab", "c"], column_widths) == "ab  c"
    assert divider(column_widths) == "--  -"
