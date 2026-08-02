"""Working with spans of seconds."""

from typing import List, Tuple

from intervals.units import from_parts, to_parts

Span = Tuple[int, int]


def length(span: Span) -> int:
    """The number of seconds a (start, end) span covers."""
    start, end = span
    return max(0, end - start)


def overlaps(left: Span, right: Span) -> bool:
    """True when two spans share at least one second."""
    return left[0] < right[1] and right[0] < left[1]


def merge(spans: List[Span]) -> List[Span]:
    """Combine overlapping and touching spans, returned in start order."""
    ordered = sorted(spans)
    merged: List[Span] = []
    for span in ordered:
        if merged and span[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], span[1]))
        else:
            merged.append(span)
    return merged


def total_length(spans: List[Span]) -> int:
    """The seconds covered by `spans`, counting overlapping time once."""
    return sum(length(span) for span in merge(spans))


def as_parts(span: Span) -> Tuple[int, int, int, int]:
    """The length of `span`, broken into days, hours, minutes and seconds."""
    return to_parts(length(span))


def from_components(days: int, hours: int, minutes: int, seconds: int) -> int:
    """Build a span length from its components."""
    return from_parts((days, hours, minutes, seconds))
