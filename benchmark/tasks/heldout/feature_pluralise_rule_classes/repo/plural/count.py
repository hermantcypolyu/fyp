"""Rendering counted quantities."""

from plural.engine import pluralise


def quantify(count: int, noun: str) -> str:
    """Render `count` of `noun`, pluralising the noun when there is not exactly one.

    A count of zero takes the plural, which is what english does: "0 cats".
    """
    if count == 1:
        return f"{count} {noun}"
    return f"{count} {pluralise(noun)}"


def join_quantities(pairs, separator: str = ", ") -> str:
    """Render several (count, noun) pairs in order."""
    return separator.join(quantify(count, noun) for count, noun in pairs)
