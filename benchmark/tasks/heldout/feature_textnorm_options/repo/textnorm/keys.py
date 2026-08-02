"""Derived keys built from the normalised form."""

from typing import List

from textnorm.normalise import normalise
from textnorm.stopwords import drop_stopwords


def sort_key(text: str) -> str:
    """The normalised tokens, sorted, so word order stops mattering."""
    return " ".join(sorted(normalise(text).split(" ")))


def dedupe_tokens(text: str) -> List[str]:
    """The normalised tokens with later repeats removed, order preserved."""
    seen = set()
    out = []
    for token in normalise(text).split(" "):
        if token and token not in seen:
            seen.add(token)
            out.append(token)
    return out


def content_key(text: str) -> str:
    """A sorted key with stopwords removed, for loose matching."""
    return " ".join(sorted(drop_stopwords(dedupe_tokens(text))))
