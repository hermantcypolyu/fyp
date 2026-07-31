"""Visible smoke tests - the agent may run these via run_tests().

Deliberately weak. They confirm the package still imports and that limiting and keying
still broadly work, so the agent can catch itself breaking the module outright. They say
nothing about buckets or refills, because run_tests() must never hand over the grading
signal (AGENT_BUILD_SPEC 5.5).
"""

from src import FixedWindowLimiter, KeyedLimiter


class StoppedClock:
    """Never moves. Enough to construct a limiter without importing time."""

    def __call__(self) -> float:
        return 0.0


def test_a_limiter_grants_then_refuses():
    gate = FixedWindowLimiter(limit=1, window_s=60.0, clock=StoppedClock())
    assert gate.allow() is True
    assert gate.allow() is False


def test_two_keys_do_not_share_an_allowance():
    clock = StoppedClock()
    keyed = KeyedLimiter(lambda: FixedWindowLimiter(limit=1, window_s=60.0, clock=clock))
    assert keyed.allow("one") is True
    assert keyed.allow("two") is True
