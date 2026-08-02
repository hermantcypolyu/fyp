"""Hidden grading suite: composing retry conditions with ``|`` and ``&``.

Each test grades one observable consequence of the composition operators, so that a
partial fix scores between the floor and one. Several tests exercise combinations that
have always worked; they are here so that a change which repairs plain-callable
composition by breaking strategy-to-strategy composition does not score full marks.
"""

import pytest

import tenacity
from tenacity.retry_state import RetryCallState


def state_returning(value, attempt_number=1):
    """A call state whose attempt returned ``value`` without raising."""
    call_state = RetryCallState(attempt_number=attempt_number)
    call_state.set_result(value)
    return call_state


def state_raising(exc, attempt_number=1):
    """A call state whose attempt raised ``exc``."""
    call_state = RetryCallState(attempt_number=attempt_number)
    call_state.set_exception(exc)
    return call_state


def early_attempt(retry_state):
    """A plain-callable retry condition: retry only on the first two attempts."""
    return retry_state.attempt_number < 3


def never(retry_state):
    """A plain-callable retry condition that never asks for a retry."""
    return False


# --------------------------------------------------------------------------------------
# Combinations that were never broken. These pin the surrounding behaviour down.
# --------------------------------------------------------------------------------------


def test_retry_any_accepts_a_plain_callable_when_constructed_directly():
    combined = tenacity.retry_any(tenacity.retry_if_result(lambda r: r == 7), early_attempt)
    assert combined(state_returning(0, attempt_number=1)) is True
    assert combined(state_returning(0, attempt_number=9)) is False


def test_retry_all_accepts_a_plain_callable_when_constructed_directly():
    combined = tenacity.retry_all(tenacity.retry_if_result(lambda r: r == 7), early_attempt)
    assert combined(state_returning(7, attempt_number=1)) is True
    assert combined(state_returning(7, attempt_number=9)) is False


def test_or_between_two_strategies_still_builds_a_retry_any():
    left = tenacity.retry_if_result(lambda r: r == 7)
    right = tenacity.retry_if_result(lambda r: r == 8)
    combined = left | right
    assert isinstance(combined, tenacity.retry_any)
    assert combined(state_returning(7)) is True
    assert combined(state_returning(8)) is True
    assert combined(state_returning(9)) is False


def test_and_between_two_strategies_still_builds_a_retry_all():
    left = tenacity.retry_if_result(lambda r: r > 5)
    right = tenacity.retry_if_result(lambda r: r < 10)
    combined = left & right
    assert isinstance(combined, tenacity.retry_all)
    assert combined(state_returning(7)) is True
    assert combined(state_returning(11)) is False


def test_plain_callable_on_the_left_of_or_still_works():
    combined = early_attempt | tenacity.retry_if_result(lambda r: r == 7)
    assert combined(state_returning(0, attempt_number=1)) is True
    assert combined(state_returning(7, attempt_number=9)) is True
    assert combined(state_returning(0, attempt_number=9)) is False


def test_plain_callable_on_the_left_of_and_still_works():
    combined = early_attempt & tenacity.retry_if_result(lambda r: r == 7)
    assert combined(state_returning(7, attempt_number=1)) is True
    assert combined(state_returning(7, attempt_number=9)) is False


# --------------------------------------------------------------------------------------
# A plain callable on the right-hand side.
# --------------------------------------------------------------------------------------


def test_strategy_or_plain_callable_builds_a_retry_any():
    combined = tenacity.retry_if_exception_type(IOError) | early_attempt
    assert isinstance(combined, tenacity.retry_any)


def test_strategy_or_plain_callable_retries_when_only_the_plain_side_matches():
    combined = tenacity.retry_if_exception_type(IOError) | early_attempt
    assert combined(state_returning("fine", attempt_number=1)) is True


def test_strategy_or_plain_callable_retries_when_only_the_strategy_matches():
    combined = tenacity.retry_if_exception_type(IOError) | early_attempt
    assert combined(state_raising(IOError("boom"), attempt_number=9)) is True


def test_strategy_or_plain_callable_declines_when_neither_side_matches():
    combined = tenacity.retry_if_exception_type(IOError) | early_attempt
    assert combined(state_returning("fine", attempt_number=9)) is False


def test_strategy_or_plain_callable_keeps_the_operands_in_written_order():
    strategy = tenacity.retry_if_result(lambda r: r == 7)
    combined = strategy | early_attempt
    assert list(combined.retries) == [strategy, early_attempt]


def test_strategy_and_plain_callable_builds_a_retry_all():
    combined = tenacity.retry_if_result(lambda r: r == 7) & early_attempt
    assert isinstance(combined, tenacity.retry_all)


def test_strategy_and_plain_callable_retries_only_when_both_sides_match():
    combined = tenacity.retry_if_result(lambda r: r == 7) & early_attempt
    assert combined(state_returning(7, attempt_number=1)) is True
    assert combined(state_returning(3, attempt_number=1)) is False


def test_strategy_and_plain_callable_declines_when_the_plain_side_refuses():
    combined = tenacity.retry_if_result(lambda r: r == 7) & never
    assert combined(state_returning(7, attempt_number=1)) is False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
