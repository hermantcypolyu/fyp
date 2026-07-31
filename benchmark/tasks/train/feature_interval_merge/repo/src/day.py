"""Turning wall-clock times into intervals.

Everything downstream works in minutes since midnight; this is the only module that
knows what an hour is.
"""

from __future__ import annotations

from .intervals import Interval

MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 24 * MINUTES_PER_HOUR


def at(hour: int, minute: int = 0) -> int:
    """Minutes since midnight for a wall-clock time on the same day."""
    if not 0 <= hour < 24:
        raise ValueError("hour must be in 0..23")
    if not 0 <= minute < MINUTES_PER_HOUR:
        raise ValueError("minute must be in 0..59")
    return hour * MINUTES_PER_HOUR + minute


def span(start_hour: int, end_hour: int) -> Interval:
    """The interval between two whole hours. ``end_hour`` of 24 means midnight."""
    end = MINUTES_PER_DAY if end_hour == 24 else at(end_hour)
    return Interval(at(start_hour), end)


def day_window() -> Interval:
    """Midnight to midnight - the usual bounding window for a day's bookings."""
    return Interval(0, MINUTES_PER_DAY)
