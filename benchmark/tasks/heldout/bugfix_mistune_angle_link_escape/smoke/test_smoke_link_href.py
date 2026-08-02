"""Visible smoke test: the slice imports and parses an ordinary destination."""

from mistune.helpers import parse_link_href


def test_bare_and_angle_destinations_parse():
    assert parse_link_href("/foo", 0) == ("/foo", 4)
    assert parse_link_href("<foo bar>", 0) == ("foo bar", 9)
