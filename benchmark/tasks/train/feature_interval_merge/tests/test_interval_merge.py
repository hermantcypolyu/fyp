"""Hidden verification suite for feature_interval_merge.

Never mounted into the sandbox. Fourteen independent assertions so that a partial
implementation earns partial credit (AGENT_BUILD_SPEC 5.4 rule 5).

The first six pin the interval behaviour that already exists and must not regress; the
remaining eight cover merging and the complement. The two new functions are only ever
reached as attributes of the module, so their absence shows up as failing tests rather
than as a collection error that scores zero for everything.
"""

import pytest

import src.intervals as intervals_module
from src.day import MINUTES_PER_DAY, at, day_window, span
from src.intervals import Interval, normalise, total_length

# --- existing behaviour: these pass before and after the feature lands ---------------


def test_an_empty_or_reversed_interval_is_rejected():
    with pytest.raises(ValueError):
        Interval(30, 30)
    with pytest.raises(ValueError):
        Interval(30, 10)


def test_an_interval_is_half_open():
    slot = Interval(10, 20)
    assert slot.length == 10
    assert slot.contains(10) is True
    assert slot.contains(19) is True
    assert slot.contains(20) is False
    assert slot.contains(9) is False


def test_touching_intervals_do_not_overlap():
    assert Interval(0, 10).overlaps(Interval(10, 20)) is False
    assert Interval(0, 10).overlaps(Interval(9, 20)) is True
    assert Interval(0, 30).overlaps(Interval(10, 20)) is True
    assert Interval(0, 10).overlaps(Interval(11, 20)) is False


def test_normalise_sorts_without_touching_the_input():
    given = [Interval(30, 40), Interval(0, 50), Interval(0, 10)]
    assert normalise(given) == [Interval(0, 10), Interval(0, 50), Interval(30, 40)]
    assert given == [Interval(30, 40), Interval(0, 50), Interval(0, 10)]


def test_total_length_adds_the_lengths_up_as_given():
    assert total_length([]) == 0
    assert total_length([Interval(0, 10), Interval(5, 15)]) == 20


def test_the_clock_helpers_build_day_sized_intervals():
    assert at(9, 30) == 570
    assert span(9, 17) == Interval(540, 1020)
    assert day_window() == Interval(0, MINUTES_PER_DAY)
    with pytest.raises(ValueError):
        at(24)


# --- the feature ---------------------------------------------------------------------


def test_merge_joins_overlapping_intervals():
    assert intervals_module.merge([Interval(0, 10), Interval(5, 20)]) == [Interval(0, 20)]


def test_merge_joins_intervals_that_only_touch():
    given = [Interval(0, 10), Interval(10, 20), Interval(20, 25)]
    assert intervals_module.merge(given) == [Interval(0, 25)]


def test_merge_sorts_its_results_and_keeps_real_gaps():
    given = [Interval(30, 40), Interval(0, 10), Interval(35, 50)]
    assert intervals_module.merge(given) == [Interval(0, 10), Interval(30, 50)]


def test_merge_of_nothing_is_nothing_and_the_input_is_left_alone():
    assert intervals_module.merge([]) == []
    given = [Interval(10, 20), Interval(0, 5)]
    intervals_module.merge(given)
    assert given == [Interval(10, 20), Interval(0, 5)]


def test_merge_absorbs_nested_and_duplicate_intervals():
    given = [Interval(0, 100), Interval(10, 20), Interval(0, 100), Interval(90, 95)]
    assert intervals_module.merge(given) == [Interval(0, 100)]


def test_the_complement_of_nothing_is_the_whole_window():
    window = Interval(0, 60)
    assert intervals_module.complement([], window) == [window]


def test_the_complement_clips_to_the_window_and_ignores_what_is_outside():
    window = Interval(100, 200)
    booked = [Interval(0, 50), Interval(80, 120), Interval(180, 260), Interval(300, 400)]
    assert intervals_module.complement(booked, window) == [Interval(120, 180)]


def test_a_window_covered_end_to_end_has_an_empty_complement():
    window = Interval(0, 60)
    assert intervals_module.complement([Interval(0, 60)], window) == []
    assert intervals_module.complement([Interval(0, 30), Interval(30, 60)], window) == []
    two_sittings = [Interval(0, 30), Interval(31, 60)]
    assert intervals_module.complement(two_sittings, window) == [Interval(30, 31)]
