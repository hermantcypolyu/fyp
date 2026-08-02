"""Splitting text into word tokens."""

import re
from typing import List

#: a run of letters or digits; underscores and punctuation are separators.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokenise(text: str) -> List[str]:
    """Split `text` into word tokens, discarding punctuation and whitespace."""
    return _WORD_RE.findall(text)


def is_word(text: str) -> bool:
    """True when `text` is a single token with nothing to strip."""
    return len(tokenise(text)) == 1 and tokenise(text)[0] == text
