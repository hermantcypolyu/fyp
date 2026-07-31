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
    what "hours occupied" means. For the second one, measure :func:`merge`.
    """
    return sum(interval.length for interval in intervals)


def merge(intervals: Iterable[Interval]) -> list[Interval]:
    """The fewest intervals covering exactly the same points, in ascending order.

    Touching counts as joined: ``[0, 10)`` and ``[10, 20)`` already describe ``[0, 20)``
    under a half-open convention, so keeping them apart would report a gap that has no
    minutes in it. The comparison is therefore ``<=`` and not ``<``.
    """
    merged: list[Interval] = []
    for interval in normalise(intervals):
        if merged and interval.start <= merged[-1].end:
            last = merged[-1]
            if interval.end > last.end:
                merged[-1] = Interval(last.start, interval.end)
        else:
            merged.append(interval)
    return merged


def complement(intervals: Iterable[Interval], window: Interval) -> list[Interval]:
    """The parts of ``window`` that nothing in ``intervals`` covers, left to right.

    Merging first is what keeps this honest: overlaps and adjacencies are gone by the
    time the cursor walks the window, so no gap of zero length can ever be emitted -
    which matters because :class:`Interval` would refuse to represent one anyway.
    """
    gaps: list[Interval] = []
    cursor = window.start
    for interval in merge(intervals):
        if interval.end <= window.start or interval.start >= window.end:
            continue
        if interval.start > cursor:
            gaps.append(Interval(cursor, interval.start))
        cursor = max(cursor, interval.end)
        if cursor >= window.end:
            break
    if cursor < window.end:
        gaps.append(Interval(cursor, window.end))
    return gaps
