"""A trimmed slice of mistune carrying only the link-destination parser."""

from .helpers import parse_link, parse_link_href, parse_link_label, parse_link_text, parse_link_title

__all__ = [
    "parse_link",
    "parse_link_href",
    "parse_link_label",
    "parse_link_text",
    "parse_link_title",
]
