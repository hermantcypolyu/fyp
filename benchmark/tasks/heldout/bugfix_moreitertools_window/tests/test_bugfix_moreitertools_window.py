from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "repo"))

from windowing import window


def test_window_returns_empty_for_too_large_size():
    assert window([1, 2], 3) == []


def test_window_returns_singletons_for_size_one():
    assert window([1, 2, 3], 1) == [(1,), (2,), (3,)]


def test_window_returns_expected_sliding_tuples():
    assert window([1, 2, 3, 4], 2) == [(1, 2), (2, 3), (3, 4)]
