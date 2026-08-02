"""Visible smoke test: the library totals a ledger."""

from tallyman import parse_lines, totals


def test_totals_by_category():
    assert totals(parse_lines(["food,12.50", "food,7.00"])) == {"food": 19.5}
