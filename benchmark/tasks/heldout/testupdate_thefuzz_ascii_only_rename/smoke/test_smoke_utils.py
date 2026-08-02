"""Visible smoke test: the library slice imports and normalises a string."""

from thefuzz import utils


def test_full_process_normalises():
    assert utils.full_process("  New York Mets  ") == "new york mets"
