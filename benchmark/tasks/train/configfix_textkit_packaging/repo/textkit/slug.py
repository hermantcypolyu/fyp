"""URL-safe slug generation."""

from __future__ import annotations

import re
import unicodedata

__all__ = ["slugify"]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(text: str, max_length: int | None = None, separator: str = "-") -> str:
    """Reduce `text` to a lowercase, URL-safe slug.

    Accents are folded to ASCII, runs of anything else collapse to a single separator,
    and the result never starts or ends with one. When `max_length` is given the slug is
    truncated on a separator boundary so words are not cut in half.

    >>> slugify("Crème Brûlée, please!")
    'creme-brulee-please'
    >>> slugify("the quick brown fox", max_length=12)
    'the-quick'
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")
    if not separator:
        raise ValueError("separator must not be empty")

    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = _NON_ALNUM.sub(separator, folded.lower()).strip(separator)

    if max_length is not None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if len(slug) > max_length:
            cut = slug[:max_length]
            head, sep, _ = cut.rpartition(separator)
            slug = (head if sep else cut).strip(separator)
    return slug
