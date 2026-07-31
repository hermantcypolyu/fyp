"""Counting and rendering, kept separate from argument parsing so both commands share it."""

from __future__ import annotations

from .parser import LEVELS, Record


def count_levels(records) -> dict[str, int]:
    """How many records carry each level. Levels with no records are omitted."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record.level] = counts.get(record.level, 0) + 1
    return {level: counts[level] for level in LEVELS if level in counts}


def format_counts(counts: dict[str, int]) -> str:
    """One `LEVEL  count` line per level, widest level name setting the column."""
    if not counts:
        return "no records"
    width = max(len(level) for level in counts)
    return "\n".join(f"{level:<{width}}  {count}" for level, count in counts.items())


def render(record: Record) -> str:
    """Canonical single-line rendering: fixed-width level, single spaces."""
    return f"{record.stamp} {record.level:<8} {record.message}"


def tidy(records, keep: set[str]) -> list[str]:
    """Render the records whose level is in `keep`, in the order they arrived."""
    return [render(record) for record in records if record.level in keep]
