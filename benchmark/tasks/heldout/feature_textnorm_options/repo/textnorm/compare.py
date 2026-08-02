"""Comparisons built on top of the normaliser."""

from typing import Set

from textnorm.normalise import normalise


def same_normalised(left: str, right: str) -> bool:
    """True when two strings normalise to the same value."""
    return normalise(left) == normalise(right)


def token_set(text: str) -> Set[str]:
    """The distinct normalised tokens of `text`."""
    normalised = normalise(text)
    return set(normalised.split(" ")) - {""}


def overlap(left: str, right: str) -> float:
    """The Jaccard overlap of two strings' normalised token sets.

    Two empty strings overlap completely, which keeps the result defined for
    every input rather than raising on the empty case.
    """
    a, b = token_set(left), token_set(right)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)
