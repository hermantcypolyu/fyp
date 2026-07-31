"""Hidden verification suite for feature_ratelimit_token_bucket.

Never mounted into the sandbox. Fourteen independent assertions so that a partial
implementation earns partial credit (AGENT_BUILD_SPEC 5.4 rule 5).

The first six pin the limiting behaviour that already exists and must not regress; the
remaining eight cover the token bucket. ``TokenBucket`` is only ever reached as an
attribute of the module, never imported by name, so that its absence shows up as eight
failing tests rather than as a collection error that scores zero for everything.

Time is supplied by :class:`FakeClock`. Nothing here imports :mod:`time`.
"""

import pytest

import src.limiter as limiter_module
from src.keyed import KeyedLimiter
from src.limiter import FixedWindowLimiter


class FakeClock:
    """A clock the test moves by hand."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# --- existing behaviour: these pass before and after the feature lands ---------------


def test_the_fixed_window_grants_exactly_its_limit():
    clock = FakeClock()
    gate = FixedWindowLimiter(limit=3, window_s=10.0, clock=clock)
    assert [gate.allow(), gate.allow(), gate.allow()] == [True, True, True]
    assert gate.allow() is False
    assert gate.remaining == 0


def test_the_fixed_window_starts_again_once_the_window_has_passed():
    clock = FakeClock()
    gate = FixedWindowLimiter(limit=2, window_s=10.0, clock=clock)
    assert gate.allow(2) is True
    clock.advance(9.0)
    assert gate.allow() is False
    clock.advance(1.0)
    assert gate.allow() is True
    assert gate.remaining == 1


def test_remaining_counts_down_as_units_are_spent():
    clock = FakeClock()
    gate = FixedWindowLimiter(limit=5, window_s=1.0, clock=clock)
    assert gate.limit == 5
    assert gate.remaining == 5
    assert gate.allow(2) is True
    assert gate.remaining == 3


def test_the_fixed_window_rejects_impossible_request_sizes():
    gate = FixedWindowLimiter(limit=2, window_s=1.0, clock=FakeClock())
    with pytest.raises(ValueError):
        gate.allow(0)
    with pytest.raises(ValueError):
        gate.allow(3)


def test_the_fixed_window_rejects_a_nonsense_configuration():
    with pytest.raises(ValueError):
        FixedWindowLimiter(limit=0, window_s=1.0, clock=FakeClock())
    with pytest.raises(ValueError):
        FixedWindowLimiter(limit=1, window_s=0.0, clock=FakeClock())


def test_each_key_is_limited_on_its_own_budget():
    clock = FakeClock()
    keyed = KeyedLimiter(lambda: FixedWindowLimiter(limit=1, window_s=10.0, clock=clock))
    assert keyed.allow("alice") is True
    assert keyed.allow("bob") is True
    assert keyed.allow("alice") is False
    assert len(keyed) == 2
    keyed.forget("alice")
    assert keyed.allow("alice") is True


# --- the feature ---------------------------------------------------------------------


def test_a_new_bucket_starts_full():
    bucket = limiter_module.TokenBucket(capacity=4, refill_per_s=1.0, clock=FakeClock())
    level = bucket.tokens
    assert isinstance(level, float)
    assert level == pytest.approx(4.0)


def test_each_grant_spends_a_token_and_an_empty_bucket_refuses():
    bucket = limiter_module.TokenBucket(capacity=2, refill_per_s=1.0, clock=FakeClock())
    assert bucket.allow() is True
    assert bucket.allow() is True
    assert bucket.tokens == pytest.approx(0.0)
    assert bucket.allow() is False
    assert bucket.tokens == pytest.approx(0.0)


def test_a_partial_refill_between_two_calls_is_carried():
    clock = FakeClock()
    bucket = limiter_module.TokenBucket(capacity=4, refill_per_s=2.0, clock=clock)
    assert bucket.allow(4) is True
    clock.advance(0.25)
    assert bucket.allow() is False
    assert bucket.tokens == pytest.approx(0.5)
    clock.advance(0.25)
    assert bucket.tokens == pytest.approx(1.0)
    assert bucket.allow() is True


def test_the_bucket_never_fills_past_its_capacity():
    clock = FakeClock()
    bucket = limiter_module.TokenBucket(capacity=3, refill_per_s=1.0, clock=clock)
    assert bucket.allow(3) is True
    clock.advance(1000.0)
    assert bucket.tokens == pytest.approx(3.0)
    assert bucket.allow(3) is True
    assert bucket.allow() is False


def test_a_multi_token_request_is_all_or_nothing():
    bucket = limiter_module.TokenBucket(capacity=5, refill_per_s=1.0, clock=FakeClock())
    assert bucket.allow(4) is True
    assert bucket.allow(2) is False
    assert bucket.tokens == pytest.approx(1.0)
    assert bucket.allow(1) is True


def test_the_bucket_rejects_impossible_sizes_and_configurations():
    clock = FakeClock()
    bucket = limiter_module.TokenBucket(capacity=2, refill_per_s=1.0, clock=clock)
    with pytest.raises(ValueError):
        bucket.allow(0)
    with pytest.raises(ValueError):
        bucket.allow(3)
    with pytest.raises(ValueError):
        limiter_module.TokenBucket(capacity=0, refill_per_s=1.0, clock=clock)
    with pytest.raises(ValueError):
        limiter_module.TokenBucket(capacity=1, refill_per_s=-1.0, clock=clock)


def test_the_token_level_is_read_only():
    bucket = limiter_module.TokenBucket(capacity=2, refill_per_s=1.0, clock=FakeClock())
    with pytest.raises(AttributeError):
        bucket.tokens = 99.0
    assert bucket.tokens == pytest.approx(2.0)


def test_a_clock_that_goes_backwards_neither_adds_nor_removes_tokens():
    clock = FakeClock(start=100.0)
    bucket = limiter_module.TokenBucket(capacity=4, refill_per_s=1.0, clock=clock)
    assert bucket.allow(4) is True
    clock.now = 90.0
    assert bucket.tokens == pytest.approx(0.0)
    assert bucket.allow() is False
    clock.now = 101.0
    assert bucket.tokens == pytest.approx(1.0)
