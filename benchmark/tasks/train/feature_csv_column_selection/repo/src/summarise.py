"""Summarising tabular data that has already been parsed into rows.

A "row" is a mapping from column name to cell value. Rows are allowed to be ragged: a
column may be absent from some of them, and a cell may be blank. Nothing in here reads
a file, a clock or a socket, so every function is cheap to test.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

Row = Mapping[str, Any]

#: Cell values that count as "not filled in". Blank strings come out of CSV; None out
#: of JSON and out of a database driver.
MISSING: tuple[Any, ...] = (None, "")


def row_count(rows: Sequence[Row]) -> int:
    """How many rows the table has."""
    return len(rows)


def column_names(rows: Sequence[Row]) -> list[str]:
    """Every column appearing in at least one row, in the order first seen."""
    seen: list[str] = []
    for row in rows:
        for name in row:
            if name not in seen:
                seen.append(name)
    return seen


def values_of(rows: Sequence[Row], column: str) -> list[Any]:
    """The filled-in cells of ``column``, in row order.

    Raises :class:`KeyError` when no row carries that column at all: asking about a
    column that does not exist is a caller bug, not an empty result.
    """
    if column not in column_names(rows):
        raise KeyError(column)
    return [row[column] for row in rows if column in row and row[column] not in MISSING]


def count_by(rows: Sequence[Row], column: str) -> dict[Any, int]:
    """Tally the filled-in values of ``column``, keyed in first-seen order.

    >>> count_by([{"c": "a"}, {"c": "b"}, {"c": "a"}], "c")
    {'a': 2, 'b': 1}
    """
    counts: dict[Any, int] = {}
    for value in values_of(rows, column):
        counts[value] = counts.get(value, 0) + 1
    return counts
