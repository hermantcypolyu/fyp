"""Applying the rules to a word."""

from plural.rules import SUFFIX_RULES

#: Words whose plural no rule can derive. Checked before anything else.
IRREGULAR = {
    "child": "children",
    "person": "people",
    "mouse": "mice",
    "goose": "geese",
    "tooth": "teeth",
    "foot": "feet",
    "man": "men",
    "woman": "women",
}

#: Words that are their own plural.
UNCOUNTABLE = {"sheep", "series", "species", "fish", "information", "equipment"}

#: `-f`/`-fe` words that keep a plain `s` instead of taking `-ves`.
VES_EXCEPTIONS = {"roof", "chief", "belief", "cliff"}


def _restore_case(word: str, plural: str) -> str:
    """Carry a leading capital across to the derived form."""
    if word[:1].isupper():
        return plural[:1].upper() + plural[1:]
    return plural


def pluralise(word: str) -> str:
    """Return the plural of `word`.

    Irregular forms and uncountable nouns are resolved first, then the
    ``-f``/``-fe`` ending, then the generic suffix rules; anything left takes a
    plain trailing ``s``. A leading capital is preserved.
    """
    if not word:
        return word

    lowered = word.lower()

    if lowered in UNCOUNTABLE:
        return word
    if lowered in IRREGULAR:
        return _restore_case(word, IRREGULAR[lowered])

    if lowered not in VES_EXCEPTIONS:
        if lowered.endswith("fe"):
            return word[:-2] + "ves"
        if lowered.endswith("f"):
            return word[:-1] + "ves"

    for rule in SUFFIX_RULES:
        if rule.matches(word):
            return rule.apply(word)
    return word + "s"
