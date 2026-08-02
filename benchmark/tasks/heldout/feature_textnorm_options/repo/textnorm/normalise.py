"""The normaliser itself."""

from textnorm.tokens import tokenise


def normalise(text: str) -> str:
    """Collapse `text` to lowercase words joined by single spaces.

    Punctuation and repeated whitespace are discarded, so
    ``"  Hello,   World! "`` becomes ``"hello world"``.
    """
    return " ".join(tokenise(text.lower()))
