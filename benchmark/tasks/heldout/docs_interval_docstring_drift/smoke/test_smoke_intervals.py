"""Visible smoke test: the library parses and renders a duration."""

from intervals import format_duration, parse_duration


def test_round_trip():
    assert parse_duration("2h30m") == 9000
    assert format_duration(9000) == "2.5h"
