"""Hidden verification for `feature_pluralise_rule_classes`.

Three classes of word that no `SuffixRule` can express are added to the engine, and they
are graded separately because they are independently implementable: an agent can land the
irregular table without the uncountables, or the `-ves` ending without the exception list,
and score accordingly.

Two of the floor tests are load-bearing rather than decorative. `roof`/`chief`/`belief`/
`cliff` already pluralise correctly today, and an agent that adds the `-f` to `-ves` rule
without the documented exception list *breaks* them - so that test can go from passing to
failing, which is exactly the regression the exception list exists to prevent. The generic
suffix rules are pinned for the same reason: the new classes are resolved ahead of them and
must not swallow the words they already handle.
"""

import pytest

from plural import detect, pluralise
from plural.count import quantify
from plural.rules import SUFFIX_RULES
from plural.singular import singularise

# --- irregular forms --------------------------------------------------------------------


def test_common_irregulars_are_looked_up():
    assert pluralise("child") == "children"
    assert pluralise("person") == "people"
    assert pluralise("mouse") == "mice"


def test_vowel_change_irregulars_are_looked_up():
    assert pluralise("goose") == "geese"
    assert pluralise("tooth") == "teeth"
    assert pluralise("foot") == "feet"


def test_man_and_woman_are_irregular():
    assert pluralise("man") == "men"
    assert pluralise("woman") == "women"


# --- uncountable nouns ------------------------------------------------------------------


def test_uncountable_nouns_are_their_own_plural():
    assert pluralise("sheep") == "sheep"
    assert pluralise("series") == "series"
    assert pluralise("species") == "species"


def test_uncountable_mass_nouns_are_unchanged():
    assert pluralise("fish") == "fish"
    assert pluralise("information") == "information"
    assert pluralise("equipment") == "equipment"


# --- the -f / -fe ending ----------------------------------------------------------------


def test_f_ending_becomes_ves():
    assert pluralise("leaf") == "leaves"
    assert pluralise("wolf") == "wolves"


def test_fe_ending_becomes_ves():
    assert pluralise("knife") == "knives"
    assert pluralise("life") == "lives"


# --- capitalisation ---------------------------------------------------------------------


def test_a_leading_capital_is_preserved():
    assert pluralise("Child") == "Children"
    assert pluralise("Leaf") == "Leaves"
    assert pluralise("Sheep") == "Sheep"


# --- floor: behaviour that must survive the new rule classes ----------------------------


def test_the_documented_ves_exceptions_keep_a_plain_s():
    # these pass today; adding -ves without the exception list breaks them
    assert pluralise("roof") == "roofs"
    assert pluralise("chief") == "chiefs"
    assert pluralise("belief") == "beliefs"
    assert pluralise("cliff") == "cliffs"


def test_consonant_y_rule_is_unchanged():
    assert pluralise("party") == "parties"
    assert pluralise("city") == "cities"


def test_vowel_y_still_takes_a_plain_s():
    assert pluralise("day") == "days"
    assert pluralise("key") == "keys"


def test_sibilant_rules_are_unchanged():
    assert pluralise("bus") == "buses"
    assert pluralise("box") == "boxes"
    assert pluralise("church") == "churches"
    assert pluralise("dish") == "dishes"


def test_the_default_rule_is_unchanged():
    assert pluralise("cat") == "cats"
    assert pluralise("table") == "tables"


def test_the_generic_rule_table_is_untouched():
    assert [(rule.suffix, rule.replacement) for rule in SUFFIX_RULES] == [
        ("y", "ies"), ("s", "ses"), ("x", "xes"),
        ("z", "zes"), ("ch", "ches"), ("sh", "shes"),
    ]


def test_helpers_and_the_empty_string_still_behave():
    assert pluralise("") == ""
    assert detect.ends_with_sibilant("bus")
    assert detect.ends_with_consonant_y("party")
    assert not detect.ends_with_consonant_y("day")


def test_the_reverse_rules_are_unaffected():
    assert singularise("parties") == "party"
    assert singularise("buses") == "bus"
    assert singularise("churches") == "church"
    assert singularise("cats") == "cat"


def test_counted_quantities_use_the_new_rule_classes():
    # quantify() defers to pluralise(), so the irregulars must reach it too
    assert quantify(1, "child") == "1 child"
    assert quantify(3, "child") == "3 children"
    assert quantify(2, "sheep") == "2 sheep"


def test_counted_quantities_still_use_the_generic_rules():
    assert quantify(0, "cat") == "0 cats"
    assert quantify(2, "party") == "2 parties"
