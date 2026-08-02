"""Visible smoke checks. Shallow by design - they confirm the package imports and that a
single retry condition still reads an outcome, and say nothing about how conditions are
combined.
"""

import tenacity


def test_package_imports_and_a_single_condition_reads_a_result():
    call_state = tenacity.RetryCallState()
    call_state.set_result(3)
    assert tenacity.retry_if_result(lambda r: r == 3)(call_state) is True


def test_the_constant_conditions_are_available():
    call_state = tenacity.RetryCallState()
    call_state.set_result(None)
    assert tenacity.retry_always(call_state) is True
    assert tenacity.retry_never(call_state) is False
