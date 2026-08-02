"""Visible smoke test: the generic suffix rules work."""

from plural import pluralise


def test_generic_rules():
    assert pluralise("party") == "parties"
    assert pluralise("bus") == "buses"
