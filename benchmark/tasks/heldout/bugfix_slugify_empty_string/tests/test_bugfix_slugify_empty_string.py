import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo"))

from slugify_utils import slugify


def test_empty_string_returns_empty_slug():
    assert slugify("") == ""


def test_whitespace_only_returns_empty_slug():
    assert slugify("   ") == ""


def test_none_returns_empty_slug():
    assert slugify(None) == ""


def test_regular_text_is_normalized():
    assert slugify("Hello World") == "hello-world"
