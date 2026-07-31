"""Parsing for the one log format logtidy understands.

    2024-03-11T08:15:04 WARNING disk usage at 91%

Lines that do not match are not an error - real log files are full of tracebacks and
banner text - so they come back as `None` and the caller decides what to do with them.
"""

from __future__ import annotations

import re
from typing import NamedTuple

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

_LINE = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"\s+(?P<level>[A-Z]+)"
    r"\s+(?P<message>.*)$"
)


class Record(NamedTuple):
    """One parsed log line."""

    stamp: str
    level: str
    message: str


def parse_line(line: str) -> Record | None:
    """Return the parsed record, or None if the line is not in the expected format."""
    match = _LINE.match(line.strip())
    if match is None:
        return None
    level = match.group("level")
    if level not in LEVELS:
        return None
    return Record(match.group("stamp"), level, match.group("message").strip())


def parse_lines(lines) -> list[Record]:
    """Parse an iterable of raw lines, silently dropping the ones that do not match."""
    out = []
    for line in lines:
        record = parse_line(line)
        if record is not None:
            out.append(record)
    return out


def at_least(level: str) -> set[str]:
    """The set of levels at or above `level`."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; expected one of {', '.join(LEVELS)}")
    return set(LEVELS[LEVELS.index(level):])
