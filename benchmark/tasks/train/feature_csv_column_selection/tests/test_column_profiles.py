"""Hidden verification suite for feature_csv_column_selection.

Never mounted into the sandbox. Thirteen independent assertions so that a partial
implementation earns partial credit (AGENT_BUILD_SPEC 5.4 rule 5).

The first five pin the counting behaviour that already exists and must not regress; the
remaining eight cover the per-column profile. `src.summarise` is imported as a module
rather than by name, so that the missing function shows up as eight failures rather
than as a collection error.
"""

import pytest

import src.summarise as summarise
from src.reader import read_rows

ROWS = [
    {"city": "leeds", "temp": "12", "note": "dry"},
    {"city": "york", "temp": "18", "note": "wet"},
    {"city": "leeds", "temp": "9", "note": ""},
    {"city": "hull", "temp": "21", "note": "dry"},
]


# --- existing behaviour: these pass before and after the feature lands ---------------


def test_row_count_counts_the_rows():
    assert summarise.row_count(ROWS) == 4
    assert summarise.row_count([]) == 0


def test_column_names_come_out_in_first_seen_order():
    ragged = [{"b": "1"}, {"a": "2", "b": "3"}, {"c": "4"}]
    assert summarise.column_names(ragged) == ["b", "a", "c"]


def test_count_by_tallies_the_filled_in_values():
    assert summarise.count_by(ROWS, "city") == {"leeds": 2, "york": 1, "hull": 1}
    assert summarise.count_by(ROWS, "note") == {"dry": 2, "wet": 1}


def test_count_by_rejects_a_column_that_is_not_there():
    with pytest.raises(KeyError):
        summarise.count_by(ROWS, "rainfall")


def test_read_rows_parses_a_csv_document():
    assert read_rows("city,temp\nleeds,12\nyork,18\n") == [
        {"city": "leeds", "temp": "12"},
        {"city": "york", "temp": "18"},
    ]


# --- the feature ---------------------------------------------------------------------


def test_a_numeric_column_reports_count_min_max_and_mean():
    assert summarise.describe(ROWS, ["temp"]) == {"temp": {"count": 4, "min": 9.0, "max": 21.0, "mean": 15.0}}


def test_the_mean_is_rounded_to_four_decimal_places():
    rows = [{"v": "1"}, {"v": "2"}, {"v": "2"}]
    assert summarise.describe(rows)["v"]["mean"] == 1.6667


def test_a_text_column_reports_count_and_the_most_common_value():
    assert summarise.describe(ROWS, ["city"]) == {"city": {"count": 4, "most_common": "leeds"}}


def test_a_tie_for_most_common_goes_to_the_earliest_value():
    rows = [{"c": "b"}, {"c": "a"}, {"c": "a"}, {"c": "b"}]
    assert summarise.describe(rows, ["c"]) == {"c": {"count": 4, "most_common": "b"}}


def test_only_the_requested_columns_come_back_and_in_that_order():
    assert list(summarise.describe(ROWS, ["note", "city"])) == ["note", "city"]
    assert list(summarise.describe(ROWS)) == ["city", "temp", "note"]


def test_missing_and_blank_cells_are_left_out_of_the_statistics():
    rows = [{"n": "4"}, {"n": ""}, {}, {"n": "6"}]
    assert summarise.describe(rows, ["n"]) == {"n": {"count": 2, "min": 4.0, "max": 6.0, "mean": 5.0}}


def test_describe_rejects_a_column_that_is_not_there():
    with pytest.raises(KeyError):
        summarise.describe(ROWS, ["rainfall"])


def test_a_wholly_blank_column_is_reported_as_categorical():
    rows = [{"x": "", "y": "1"}, {"x": "", "y": "2"}]
    assert summarise.describe(rows, ["x"]) == {"x": {"count": 0, "most_common": None}}
