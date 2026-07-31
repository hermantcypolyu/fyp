"""Summarising tabular data that has already been parsed into rows.

A "row" is a mapping from column name to cell value. Rows are allowed to be ragged: a
column may be absent from some of them, and a cell may be blank. Nothing in here reads
a file, a clock or a socket, so every function is cheap to test.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

Row = Mapping[str, Any]

#: Cell values that count as "not filled in". Blank strings come out of CSV; None out
#: of JSON and out of a database driver.
MISSING: tuple[Any, ...] = (None, "")

#: Decimal places the mean is reported to. Reports are read by people, not by solvers.
MEAN_PLACES = 4


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


def describe(rows: Sequence[Row], columns: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    """Profile a selection of columns, one entry per column.

    ``columns`` picks the subset to report on and fixes the order of the result;
    ``None`` means every column, in :func:`column_names` order. A requested column that
    appears in no row raises :class:`KeyError`, exactly as :func:`values_of` does.

    A column whose filled-in cells all parse as numbers is reported as
    ``{"count", "min", "max", "mean"}``; anything else - including a column with no
    filled-in cells at all - as ``{"count", "most_common"}``.

    >>> describe([{"n": "1"}, {"n": "2"}])
    {'n': {'count': 2, 'min': 1.0, 'max': 2.0, 'mean': 1.5}}
    """
    known = column_names(rows)
    wanted = list(known) if columns is None else list(columns)

    profile: dict[str, dict[str, Any]] = {}
    for column in wanted:
        if column not in known:
            raise KeyError(column)
        values = values_of(rows, column)
        numbers = _as_floats(values)
        if numbers is None:
            profile[column] = {"count": len(values), "most_common": _most_common(values)}
        else:
            profile[column] = {
                "count": len(values),
                "min": min(numbers),
                "max": max(numbers),
                "mean": round(sum(numbers) / len(numbers), MEAN_PLACES),
            }
    return profile


def _as_floats(values: Sequence[Any]) -> list[float] | None:
    """Every value as a float, or None if the column is empty or not wholly numeric."""
    if not values:
        return None
    numbers: list[float] = []
    for value in values:
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            return None
    return numbers


def _most_common(values: Sequence[Any]) -> Any | None:
    """The most frequent value, ties going to the earliest. None when there are none."""
    best: Any | None = None
    best_count = 0
    counts: dict[Any, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    for value, count in counts.items():  # insertion order == first appearance
        if count > best_count:
            best, best_count = value, count
    return best
