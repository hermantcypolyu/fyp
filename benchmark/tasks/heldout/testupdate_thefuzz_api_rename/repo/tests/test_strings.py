from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repo"))

from compare import compare_text


def test_compare_is_case_insensitive():
    assert compare_text("Alpha", "alpha") == 1


def test_compare_detects_difference():
    assert compare_text("Alpha", "Beta") == 0
