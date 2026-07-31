"""Visible smoke tests - the agent may run these via run_tests().

Deliberately weak. They confirm the package still imports and that building an interval
from clock times still works, so the agent can catch itself breaking the module
outright. They say nothing about coverage or free time, because run_tests() must never
hand over the grading signal (AGENT_BUILD_SPEC 5.5).
"""

from src import Interval, span, total_length


def test_a_span_is_built_from_clock_hours():
    assert span(9, 17) == Interval(540, 1020)
    assert span(9, 17).length == 480


def test_lengths_still_add_up():
    assert total_length([Interval(0, 15), Interval(45, 60)]) == 30
