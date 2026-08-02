from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "repo"))

from compare import compare


def test_compare_is_case_insensitive():
    assert compare("Alpha", "alpha") == 1


def test_compare_detects_difference():
    assert compare("Alpha", "Beta") == 0
