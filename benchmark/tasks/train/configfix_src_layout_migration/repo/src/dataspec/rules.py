"""Reusable value rules.

A rule is a callable taking one value and returning either `None`, meaning the value is
acceptable, or a short message naming what is wrong with it. Self-contained: nothing here
imports a sibling module.
"""

from __future__ import annotations

import re


def min_length(length: int):
    """Reject values shorter than `length`."""

    def rule(value):
        if len(value) < length:
            return f"is shorter than {length}"
        return None

    return rule


def one_of(*allowed):
    """Reject values outside `allowed`."""
    choices = tuple(allowed)

    def rule(value):
        if value not in choices:
            return f"is not one of {', '.join(str(choice) for choice in choices)}"
        return None

    return rule


def matches(pattern: str):
    """Reject values whose string form does not fully match `pattern`."""
    compiled = re.compile(pattern)

    def rule(value):
        if compiled.fullmatch(str(value)) is None:
            return f"does not match {pattern}"
        return None

    return rule
