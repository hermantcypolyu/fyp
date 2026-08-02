"""The normaliser itself."""

from textnorm.accents import fold_accents
from textnorm.tokens import tokenise


def normalise(
    text: str,
    *,
    fold: bool = False,
    keep_case: bool = False,
    separator: str = " ",
) -> str:
    """Collapse `text` to words joined by `separator`.

    Punctuation and repeated whitespace are discarded, so
    ``"  Hello,   World! "`` becomes ``"hello world"``.

    :param fold: replace accented characters with their unaccented equivalents
    :param keep_case: leave the original case alone instead of lowercasing
    :param separator: what to join the surviving tokens with
    """
    working = text if keep_case else text.lower()
    if fold:
        working = fold_accents(working)
    return separator.join(tokenise(working))
