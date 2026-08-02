"""Unit table and the arithmetic shared by the parser and the formatter."""

from typing import Dict, Tuple

#: suffix -> seconds. Ordered longest-first so rendering picks the largest unit.
UNITS: Dict[str, int] = {"d": 86400, "h": 3600, "m": 60, "s": 1}

#: the plural english names, used by the formatter's long form.
LONG_NAMES: Dict[str, str] = {"d": "days", "h": "hours", "m": "minutes", "s": "seconds"}


def largest_unit(seconds: int) -> str:
    """Return the suffix of the largest unit that divides into `seconds`."""
    for suffix, size in UNITS.items():
        if abs(seconds) >= size:
            return suffix
    return "s"


def to_parts(seconds: int) -> Tuple[int, int, int, int]:
    """Break a whole number of seconds into (days, hours, minutes, seconds)."""
    rest = int(seconds)
    days, rest = divmod(rest, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    return days, hours, minutes, secs


def from_parts(parts: Tuple[int, int, int, int]) -> int:
    """Collapse (days, hours, minutes, seconds) back into whole seconds."""
    days, hours, minutes, secs = parts
    return days * 86400 + hours * 3600 + minutes * 60 + secs
