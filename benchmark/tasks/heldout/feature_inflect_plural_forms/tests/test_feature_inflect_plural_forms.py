from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo"))

from pluralize import pluralize


def test_irregular_forms_are_supported():
    assert pluralize("child") == "children"
    assert pluralize("person") == "people"


def test_regular_suffix_rules_still_work():
    assert pluralize("party") == "parties"
    assert pluralize("bus") == "buses"


def test_additional_irregular_forms_are_supported():
    assert pluralize("mouse") == "mice"
    assert pluralize("leaf") == "leaves"
