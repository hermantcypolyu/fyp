from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo"))

from normalizer import normalize


def test_normalize_lowercases_and_collapses_whitespace():
    assert normalize("  Hello   WORLD  ") == "hello world"


def test_normalize_replaces_hyphens_with_spaces():
    assert normalize("Hello-World") == "hello world"


def test_normalize_handles_empty_input():
    assert normalize("") == ""
