"""Normalise text for comparison."""

from textnorm.accents import fold_accents
from textnorm.compare import overlap, same_normalised, token_set
from textnorm.keys import content_key, dedupe_tokens, sort_key
from textnorm.normalise import normalise
from textnorm.stopwords import drop_stopwords, is_stopword
from textnorm.tokens import is_word, tokenise

__all__ = [
    "content_key",
    "dedupe_tokens",
    "drop_stopwords",
    "fold_accents",
    "is_stopword",
    "is_word",
    "normalise",
    "overlap",
    "same_normalised",
    "sort_key",
    "token_set",
    "tokenise",
]
