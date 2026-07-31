"""Case conversions."""

from __future__ import annotations

import re

__all__ = ["title_case", "snake_case", "camel_case"]

#: Words left lowercase unless they open or close the title.
SMALL_WORDS = frozenset(
    "a an and as at but by en for if in nor of on or per the to v v. via vs vs.".split()
)

_WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[\s\-_]+")


def _words(text: str) -> list[str]:
    return [w for w in _WORD_BOUNDARY.split(text) if w]


def title_case(text: str, small_words: frozenset[str] | None = None) -> str:
    """Capitalise `text` as a title, leaving minor words lowercase in the middle.

    >>> title_case("the wind in the willows")
    'The Wind in the Willows'
    """
    small = SMALL_WORDS if small_words is None else small_words
    words = _words(text)
    if not words:
        return ""
    out = []
    for i, word in enumerate(words):
        lower = word.lower()
        first_or_last = i == 0 or i == len(words) - 1
        out.append(lower if (lower in small and not first_or_last) else lower.capitalize())
    return " ".join(out)


def snake_case(text: str) -> str:
    """>>> snake_case("HTTPResponseCode")
    'http_response_code'
    """
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return "_".join(w.lower() for w in _words(spaced))


def camel_case(text: str, upper_first: bool = False) -> str:
    """>>> camel_case("http response code")
    'httpResponseCode'
    """
    words = [w.lower() for w in _words(text)]
    if not words:
        return ""
    head = words[0].capitalize() if upper_first else words[0]
    return head + "".join(w.capitalize() for w in words[1:])
