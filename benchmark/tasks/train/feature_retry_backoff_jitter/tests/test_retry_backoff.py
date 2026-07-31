"""Hidden verification suite for feature_retry_backoff_jitter.

Never mounted into the sandbox. Fourteen independent assertions so that a partial
implementation earns partial credit (AGENT_BUILD_SPEC 5.4 rule 5).

The first six pin the fixed-delay behaviour that already exists and must not regress;
the remaining eight cover the backoff/cap/jitter feature. Everything is deterministic:
the sleeper and the random source are both injected stand-ins, so no real clock and no
real entropy is involved.
"""

import pytest

from src.retry import call_with_retry, retry


class Recorder:
    """Stands in for time.sleep and remembers every wait it was asked for."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


class FixedFraction:
    """A deterministic stand-in for random.Random: always the same point in the range."""

    def __init__(self, fraction: float) -> None:
        self.fraction = fraction
        self.calls: list[tuple[float, float]] = []

    def uniform(self, a: float, b: float) -> float:
        self.calls.append((a, b))
        return a + (b - a) * self.fraction


class Boom(RuntimeError):
    pass


def failing(times: int, value: str = "ok"):
    """A zero-argument callable that raises Boom `times` times, then returns `value`."""
    state = {"n": 0}

    def call() -> str:
        state["n"] += 1
        if state["n"] <= times:
            raise Boom(f"attempt {state['n']}")
        return value

    return call


# --- existing behaviour: these pass before and after the feature lands ---------------


def test_a_call_that_succeeds_first_time_never_sleeps():
    sleeper = Recorder()
    assert call_with_retry(failing(0), attempts=3, delay=1.0, sleeper=sleeper) == "ok"
    assert sleeper.waits == []


def test_the_final_exception_is_reraised_once_the_budget_runs_out():
    sleeper = Recorder()
    with pytest.raises(Boom):
        call_with_retry(failing(9), attempts=3, delay=1.0, sleeper=sleeper)
    assert len(sleeper.waits) == 2


def test_an_exception_outside_the_retry_list_propagates_immediately():
    sleeper = Recorder()

    def call():
        raise KeyError("not transient")

    with pytest.raises(KeyError):
        call_with_retry(call, attempts=4, delay=1.0, on=(Boom,), sleeper=sleeper)
    assert sleeper.waits == []


def test_the_default_policy_waits_a_constant_delay():
    sleeper = Recorder()
    call_with_retry(failing(3), attempts=4, delay=0.25, sleeper=sleeper)
    assert sleeper.waits == [0.25, 0.25, 0.25]


def test_impossible_attempt_and_delay_settings_are_rejected():
    with pytest.raises(ValueError):
        call_with_retry(failing(0), attempts=0)
    with pytest.raises(ValueError):
        call_with_retry(failing(0), delay=-1.0)


def test_the_decorator_keeps_the_wrapped_function_identity():
    sleeper = Recorder()

    @retry(attempts=3, delay=1.0, sleeper=sleeper)
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    assert add(2, 3) == 5
    assert add.__name__ == "add"
    assert add.__doc__ == "Add two numbers."


# --- the feature ---------------------------------------------------------------------


def test_exponential_backoff_multiplies_the_wait_each_time():
    sleeper = Recorder()
    call_with_retry(failing(3), attempts=4, delay=1.0, backoff=2.0, sleeper=sleeper)
    assert sleeper.waits == [1.0, 2.0, 4.0]


def test_max_delay_caps_the_growing_wait():
    sleeper = Recorder()
    call_with_retry(failing(4), attempts=5, delay=1.0, backoff=3.0, max_delay=5.0, sleeper=sleeper)
    assert sleeper.waits == [1.0, 3.0, 5.0, 5.0]


def test_full_jitter_draws_between_zero_and_the_planned_wait():
    rng = FixedFraction(1.0)
    call_with_retry(failing(2), attempts=3, delay=1.0, backoff=2.0, jitter=True, rng=rng, sleeper=Recorder())
    assert rng.calls == [(0.0, 1.0), (0.0, 2.0)]


def test_the_jittered_wait_is_whatever_the_injected_source_returns():
    sleeper = Recorder()
    call_with_retry(failing(2), attempts=3, delay=4.0, jitter=True, rng=FixedFraction(0.25), sleeper=sleeper)
    assert sleeper.waits == [1.0, 1.0]


def test_jitter_is_applied_to_the_already_capped_wait():
    sleeper = Recorder()
    rng = FixedFraction(1.0)
    call_with_retry(
        failing(2), attempts=3, delay=8.0, backoff=2.0, max_delay=3.0, jitter=True, rng=rng, sleeper=sleeper
    )
    assert rng.calls == [(0.0, 3.0), (0.0, 3.0)]
    assert sleeper.waits == [3.0, 3.0]


def test_a_backoff_below_one_is_a_configuration_error():
    with pytest.raises(ValueError):
        call_with_retry(failing(0), backoff=0.5)


def test_a_negative_ceiling_is_a_configuration_error():
    with pytest.raises(ValueError):
        call_with_retry(failing(0), max_delay=-1.0)


def test_the_decorator_forwards_the_new_options():
    sleeper = Recorder()

    @retry(attempts=4, delay=1.0, backoff=2.0, max_delay=3.0, sleeper=sleeper)
    def always_fails() -> str:
        raise Boom("always")

    with pytest.raises(Boom):
        always_fails()
    assert sleeper.waits == [1.0, 2.0, 3.0]
