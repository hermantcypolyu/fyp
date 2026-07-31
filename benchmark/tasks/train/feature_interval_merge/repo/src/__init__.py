"""slotwork - half-open interval arithmetic for room and resource booking."""

from .day import MINUTES_PER_DAY, MINUTES_PER_HOUR, at, day_window, span
from .intervals import Interval, normalise, total_length

__all__ = [
    "MINUTES_PER_DAY",
    "MINUTES_PER_HOUR",
    "Interval",
    "at",
    "day_window",
    "normalise",
    "span",
    "total_length",
]
