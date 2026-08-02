"""Applying the rules to a word."""

from plural.rules import SUFFIX_RULES


def pluralise(word: str) -> str:
    """Return the plural of `word`.

    The generic suffix rules are tried in order; anything they do not match
    takes a plain trailing ``s``.
    """
    if not word:
        return word
    for rule in SUFFIX_RULES:
        if rule.matches(word):
            return rule.apply(word)
    return word + "s"
