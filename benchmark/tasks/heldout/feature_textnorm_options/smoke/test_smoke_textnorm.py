"""Visible smoke test: the normaliser collapses a string."""

from textnorm import normalise


def test_default_normalisation():
    assert normalise("  Hello,   World! ") == "hello world"
