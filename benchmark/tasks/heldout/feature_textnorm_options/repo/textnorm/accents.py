"""Latin-1 and Latin Extended-A folding table."""

#: accented character -> its unaccented ASCII equivalent.
ACCENT_MAP = {
    "à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a", "ā": "a", "ă": "a", "ą": "a",
    "ç": "c", "ć": "c", "ĉ": "c", "ċ": "c", "č": "c",
    "ð": "d", "ď": "d", "đ": "d",
    "è": "e", "é": "e", "ê": "e", "ë": "e", "ē": "e", "ĕ": "e", "ė": "e", "ę": "e", "ě": "e",
    "ĝ": "g", "ğ": "g", "ġ": "g", "ģ": "g",
    "ĥ": "h", "ħ": "h",
    "ì": "i", "í": "i", "î": "i", "ï": "i", "ĩ": "i", "ī": "i", "ĭ": "i", "į": "i", "ı": "i",
    "ĵ": "j",
    "ķ": "k",
    "ĺ": "l", "ļ": "l", "ľ": "l", "ł": "l",
    "ñ": "n", "ń": "n", "ņ": "n", "ň": "n",
    "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o", "ø": "o", "ō": "o", "ŏ": "o", "ő": "o",
    "ŕ": "r", "ŗ": "r", "ř": "r",
    "ś": "s", "ŝ": "s", "ş": "s", "š": "s",
    "ţ": "t", "ť": "t", "ŧ": "t",
    "ù": "u", "ú": "u", "û": "u", "ü": "u", "ũ": "u", "ū": "u", "ŭ": "u", "ů": "u", "ű": "u",
    "ŵ": "w",
    "ý": "y", "ÿ": "y", "ŷ": "y",
    "ź": "z", "ż": "z", "ž": "z",
    "æ": "ae", "œ": "oe", "ß": "ss", "þ": "th",
}


def fold_accents(text: str) -> str:
    """Replace accented characters with their unaccented equivalents.

    Characters outside the table are passed through unchanged, and case is
    preserved by folding the lowercase form and restoring the original case.
    """
    out = []
    for char in text:
        replacement = ACCENT_MAP.get(char.lower())
        if replacement is None:
            out.append(char)
        elif char.isupper():
            out.append(replacement.capitalize() if len(replacement) > 1 else replacement.upper())
        else:
            out.append(replacement)
    return "".join(out)
