"""Cheap heuristics for inspecting a word."""

from plural.rules import VOWELS


def ends_with_sibilant(word: str) -> bool:
    """True when a word needs an ``es`` plural because of how it ends."""
    lowered = word.lower()
    return lowered.endswith(("s", "x", "z", "ch", "sh"))


def ends_with_consonant_y(word: str) -> bool:
    """True when a word ends in a consonant followed by ``y``."""
    lowered = word.lower()
    if not lowered.endswith("y") or len(lowered) < 2:
        return False
    return lowered[-2] not in VOWELS


def looks_plural(word: str) -> bool:
    """A rough guess at whether `word` is already plural.

    Deliberately conservative: it only reports True for the endings the rules
    in this package actually produce.
    """
    lowered = word.lower()
    if lowered.endswith(("ies", "ses", "xes", "zes", "ches", "shes", "ves")):
        return True
    return lowered.endswith("s") and not lowered.endswith("ss")
