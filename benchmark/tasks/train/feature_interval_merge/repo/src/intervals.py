"""Half-open time intervals, ``[start, end)``.

Half-open is the only convention that composes. Two intervals that share an endpoint are
*adjacent* rather than overlapping, so a day can be tiled without anyone having to decide
which side owns the boundary minute. Endpoints here are whole minutes since midnight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, order=True)
class Interval:
    """A half-open span ``[start, end)``. Empty and reversed spans are rejected."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"end must be > start, got [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        """How long the span is, in the units of its endpoints."""
        return self.end - self.start

    def contains(self, point: int) -> bool:
        """Is ``point`` inside? The start is in, the end is out."""
        return self.start <= point < self.end

    def overlaps(self, other: Interval) -> bool:
        """Do the two share at least one point? Merely touching does not count."""
        return self.start < other.end and other.start < self.end


def normalise(intervals: Iterable[Interval]) -> list[Interval]:
    """A new list of the same intervals, ordered by start and then by end."""
    return sorted(intervals)


def total_length(intervals: Iterable[Interval]) -> int:
    """The lengths added up exactly as given.

    This counts anything booked twice twice, which is what "hours sold" means and not
    what "hours occupied" means. There is currently no way to ask for the second one.
    """
    return sum(interval.length for interval in intervals)
