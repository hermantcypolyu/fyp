"""Common words that carry little meaning when comparing text."""

from typing import Iterable, List, Set

#: Deliberately short: only words that are noise in almost any domain.
STOPWORDS: Set[str] = {
    "a", "an", "and", "as", "at", "be", "but", "by", "for", "from", "if", "in",
    "is", "it", "of", "on", "or", "the", "to", "with",
}


def is_stopword(token: str) -> bool:
    return token.lower() in STOPWORDS


def drop_stopwords(tokens: Iterable[str]) -> List[str]:
    """Filter stopwords out of a token sequence, preserving order.

    If every token is a stopword the input is returned unchanged, so that a
    phrase made entirely of common words does not compare as empty.
    """
    tokens = list(tokens)
    kept = [token for token in tokens if not is_stopword(token)]
    return kept if kept else tokens
