"""The inverse operation: recovering a singular from a regular plural."""

from typing import List, Tuple

#: (plural ending, singular ending), longest first so `ches` beats `s`.
REVERSE_RULES: List[Tuple[str, str]] = [
    ("ches", "ch"),
    ("shes", "sh"),
    ("ies", "y"),
    ("ses", "s"),
    ("xes", "x"),
    ("zes", "z"),
]


def singularise(word: str) -> str:
    """Undo the generic suffix rules.

    Only the endings this package produces are recognised, and a word that
    matches none of them is returned unchanged. Irregular and uncountable
    nouns are outside the scope of the reverse rules.
    """
    if not word:
        return word
    lowered = word.lower()
    for plural_ending, singular_ending in REVERSE_RULES:
        if lowered.endswith(plural_ending):
            return word[: -len(plural_ending)] + singular_ending
    if lowered.endswith("s") and not lowered.endswith("ss"):
        return word[:-1]
    return word


def round_trips(word: str) -> bool:
    """True when singularising the plural of `word` gives `word` back.

    Useful as a sanity check on the rule table: every word the generic rules
    handle should survive the round trip.
    """
    from plural.engine import pluralise

    return singularise(pluralise(word)) == word
