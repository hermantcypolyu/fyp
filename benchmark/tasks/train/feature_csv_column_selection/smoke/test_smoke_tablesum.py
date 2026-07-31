"""Visible smoke tests - the agent may run these via run_tests().

Deliberately weak. They confirm the package still imports and that parsing and counting
still work, so the agent can catch itself breaking the module outright. They say nothing
about per-column profiles, because run_tests() must never hand over the grading signal
(AGENT_BUILD_SPEC 5.5).
"""

from src import column_names, count_by, read_rows, row_count


def test_a_csv_document_still_parses_into_rows():
    assert read_rows("a,b\n1,2\n") == [{"a": "1", "b": "2"}]


def test_rows_and_columns_are_still_countable():
    rows = [{"a": "1"}, {"a": "2", "b": "3"}]
    assert row_count(rows) == 2
    assert column_names(rows) == ["a", "b"]
    assert count_by(rows, "a") == {"1": 1, "2": 1}
