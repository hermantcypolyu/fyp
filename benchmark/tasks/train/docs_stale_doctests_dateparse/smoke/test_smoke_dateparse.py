"""Visible smoke checks - this is what run_tests() shows the agent.

Deliberately weak: they confirm the package still imports and that the two modules still
talk to each other. They say nothing about the documentation, because run_tests() must
never hand over the grading signal (AGENT_BUILD_SPEC 5.5).
"""

from datetime import date

from src import parse_iso_date, reporting_window, sign_off_deadline


def test_a_parsed_date_survives_a_round_trip():
    assert parse_iso_date("2024-03-05") == date(2024, 3, 5)


def test_a_monthly_window_closes_before_its_sign_off_deadline():
    start, end = reporting_window(2024, 2)
    assert start < end
    assert sign_off_deadline(end) > end
