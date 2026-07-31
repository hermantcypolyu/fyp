"""Column measurement and row rendering for plain-text tables. No dependencies."""

from __future__ import annotations


def widths(rows: list[list[str]]) -> list[int]:
    """The display width of each column: the widest cell in it."""
    if not rows:
        return []
    columns = max(len(row) for row in rows)
    return [max((len(row[i]) for row in rows if i < len(row)), default=0) for i in range(columns)]


def format_row(row: list[str], column_widths: list[int], pad: int = 2) -> str:
    """One row, each cell left-aligned in its column."""
    gap = " " * pad
    cells = [cell.ljust(column_widths[i]) for i, cell in enumerate(row)]
    return gap.join(cells).rstrip()


def divider(column_widths: list[int], pad: int = 2) -> str:
    """The dashed rule that sits under the header row."""
    return (" " * pad).join("-" * width for width in column_widths)
