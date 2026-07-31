"""Visible smoke checks: the library still imports and still parses a line.

Deliberately says nothing about console-script metadata - it passes before and after.
"""

from logtidy.parser import parse_line
from logtidy.report import count_levels


def test_parse_line_returns_none_for_junk():
    assert parse_line("not a log line at all") is None


def test_count_levels_is_empty_for_no_records():
    assert count_levels([]) == {}
