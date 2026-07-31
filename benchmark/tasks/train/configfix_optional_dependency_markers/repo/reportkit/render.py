"""Rendering a report to plain text.

The plain-text renderer is the one everybody gets: it needs nothing beyond the standard
library. The YAML, spreadsheet and fast-JSON renderers are optional and go through
`features.require` first, so that a missing extra fails with an install hint rather than
a bare ModuleNotFoundError.
"""

from __future__ import annotations

from .features import require
from .table import divider, format_row, widths


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    """A header row, a rule, then the data rows."""
    column_widths = widths([headers, *rows])
    lines = [format_row(headers, column_widths), divider(column_widths)]
    lines.extend(format_row(row, column_widths) for row in rows)
    return "\n".join(lines)


def render_report(title: str, headers: list[str], rows: list[list[str]]) -> str:
    """A titled report: title, underline, blank line, table."""
    return "\n".join([title, "=" * len(title), "", render_table(headers, rows)])


def render_yaml(headers: list[str], rows: list[list[str]]) -> str:
    """The same report as a YAML document. Needs the optional YAML extra."""
    require("yaml")
    import yaml  # noqa: PLC0415 - deliberately deferred until the extra is confirmed

    return yaml.safe_dump([dict(zip(headers, row, strict=False)) for row in rows], sort_keys=False)
