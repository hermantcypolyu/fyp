"""Turn english nouns into their plural form."""

from plural.count import join_quantities, quantify
from plural.detect import ends_with_consonant_y, ends_with_sibilant, looks_plural
from plural.engine import pluralise
from plural.rules import SUFFIX_RULES, SuffixRule
from plural.singular import round_trips, singularise

__all__ = [
    "SUFFIX_RULES",
    "SuffixRule",
    "ends_with_consonant_y",
    "ends_with_sibilant",
    "join_quantities",
    "looks_plural",
    "pluralise",
    "quantify",
    "round_trips",
    "singularise",
]
