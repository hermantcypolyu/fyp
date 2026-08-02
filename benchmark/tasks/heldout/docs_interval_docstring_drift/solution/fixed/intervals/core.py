"""Parse and render short duration strings."""

import re
from typing import Optional, Tuple

from intervals.units import LONG_NAMES, UNITS, from_parts, largest_unit, to_parts

_TOKEN_RE = re.compile(r"(\d+)\s*([dhms])")


def parse_duration(text: str) -> Optional[int]:
    """Parse a duration like ``2h30m`` into whole seconds.

    Unit suffixes are ``d``, ``h``, ``m`` and ``s``, may appear in any order,
    and may be separated by whitespace. Repeated units accumulate.

        >>> parse_duration("2h30m")
        9000
        >>> parse_duration("1d 12h")
        129600

    :param text: the duration to read
    :returns: the total in seconds, or None when no unit is recognised
    """
    matches = _TOKEN_RE.findall(text.lower())
    if not matches:
        return None
    return sum(int(count) * UNITS[suffix] for count, suffix in matches)


def format_duration(seconds: int, *, precision: int = 1) -> str:
    """Render `seconds` using the largest unit that fits.

    The value is scaled into that unit and rounded, then the bare suffix is
    appended.

        >>> format_duration(9000)
        '2.5h'
        >>> format_duration(90, precision=0)
        '2m'

    :param seconds: a whole number of seconds
    :param precision: how many decimal places to keep
    :returns: the rendered duration
    """
    suffix = largest_unit(seconds)
    scaled = seconds / UNITS[suffix]
    return f"{scaled:.{precision}f}{suffix}"


def format_long(seconds: int, *, precision: int = 1) -> str:
    """Render `seconds` using the plural english name of the largest unit.

        >>> format_long(9000)
        '2.5 hours'

    :param seconds: a whole number of seconds
    :param precision: how many decimal places to keep
    :returns: the rendered duration
    """
    suffix = largest_unit(seconds)
    scaled = seconds / UNITS[suffix]
    return f"{scaled:.{precision}f} {LONG_NAMES[suffix]}"


def split_units(seconds: int) -> Tuple[int, int, int, int]:
    """Split `seconds` into days, hours, minutes and seconds.

        >>> split_units(9000)
        (0, 2, 30, 0)

    :param seconds: a whole number of seconds
    :returns: a tuple of (days, hours, minutes, seconds)
    """
    return to_parts(seconds)


def total_seconds(parts: Tuple[int, int, int, int]) -> int:
    """Collapse a (days, hours, minutes, seconds) tuple into seconds.

        >>> total_seconds((0, 2, 30, 0))
        9000

    :param parts: the four components, largest first
    :returns: the total in seconds
    """
    return from_parts(parts)


def clamp_duration(seconds: int, low: int, high: int) -> int:
    """Constrain `seconds` to the inclusive range [`low`, `high`].

        >>> clamp_duration(100, 0, 60)
        60

    :param seconds: the value to constrain
    :param low: the lower bound
    :param high: the upper bound
    :returns: the nearest bound when `seconds` falls outside the range,
        otherwise `seconds` unchanged
    """
    if seconds < low:
        return low
    if seconds > high:
        return high
    return seconds
