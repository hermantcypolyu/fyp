"""Hidden verification for `bugfix_throttle_timedelta`.

Assertions are taken from the upstream suite at the fixing commit
(Suor/funcy @ 6c111987f25eb8371e33b71f8b76e2975c30617d, tests/test_flow.py), where
`test_throttle` is a two-way `@pytest.mark.parametrize` over `int` and
`timedelta(seconds=...)`. That parametrisation is split here into separate behaviour-sized
tests so partial credit is meaningful, and extended with the neighbouring durations that
run through the same conversion.

Every test drives `time.time` from a fixed iterator, exactly as the upstream test does, so
nothing here sleeps or reads a real clock.

Two of the passing tests exist to block plausible wrong fixes:

* `test_int_period_still_measured_with_time_time` - rewriting `throttle` to keep the
  period as a `timedelta` and compare `datetime.now()` values would make the timedelta
  cases pass while silently detaching the int and float cases from `time.time()`. This
  test pins the deadline arithmetic to `time.time()`.
* `test_two_throttled_functions_are_independent` - hoisting the deadline into a variable
  shared by the whole decorator (rather than per decorated function) is an easy way to
  "simplify" this code, and would make one function's call suppress another's.

`test_seconds_attribute_is_not_enough` and `test_period_longer_than_a_day` additionally
rule out `period.seconds`, which silently truncates sub-second and multi-day durations,
and `int(period.total_seconds())`.
"""

from datetime import timedelta

from funcy import throttle


def fake_clock(monkeypatch, timestamps):
    """Makes `time.time()` return the given values, in order."""
    ticks = iter(timestamps)
    monkeypatch.setattr('time.time', lambda: next(ticks))


# --- behaviour that already works, and must keep working --------------------------------


def test_int_period_still_measured_with_time_time(monkeypatch):
    fake_clock(monkeypatch, [0, 0.01, 1, 1.000025])
    calls = []

    @throttle(1)
    def throttled(x):
        calls.append(x)

    throttled(1)
    throttled(2)
    throttled(3)
    throttled(4)
    assert calls == [1, 3]


def test_float_period_is_accepted(monkeypatch):
    fake_clock(monkeypatch, [0, 0.4, 0.6])
    calls = []

    @throttle(0.5)
    def throttled(x):
        calls.append(x)

    throttled(1)
    throttled(2)
    throttled(3)
    assert calls == [1, 3]


def test_two_throttled_functions_are_independent(monkeypatch):
    fake_clock(monkeypatch, [0, 0, 0.5, 0.5])
    first_calls, second_calls = [], []

    @throttle(1)
    def first(x):
        first_calls.append(x)

    @throttle(1)
    def second(x):
        second_calls.append(x)

    first(1)
    second(1)
    first(2)
    second(2)
    assert first_calls == [1]
    assert second_calls == [1]


def test_suppressed_call_returns_none_and_allowed_call_returns_result(monkeypatch):
    fake_clock(monkeypatch, [0, 0.5])

    @throttle(1)
    def throttled(x):
        return x * 2

    assert throttled(3) == 6
    assert throttled(4) is None


def test_decorator_preserves_function_metadata():
    @throttle(1)
    def throttled(x):
        """Does a thing."""

    assert throttled.__name__ == 'throttled'
    assert throttled.__doc__ == 'Does a thing.'


# --- the defect -------------------------------------------------------------------------
# A period given as a timedelta has to be converted to a number of seconds before it can
# be added to a `time.time()` reading.


def test_timedelta_period_can_be_applied(monkeypatch):
    fake_clock(monkeypatch, [0, 0.01, 1, 1.000025])
    calls = []

    @throttle(timedelta(seconds=1))
    def throttled(x):
        calls.append(x)

    throttled(1)
    throttled(2)
    throttled(3)
    throttled(4)
    assert calls == [1, 3]


def test_decorating_with_a_timedelta_does_not_raise():
    @throttle(timedelta(seconds=30))
    def throttled(x):
        return x

    assert callable(throttled)


def test_subsecond_timedelta_period(monkeypatch):
    fake_clock(monkeypatch, [0, 0.2, 0.6])
    calls = []

    @throttle(timedelta(milliseconds=500))
    def throttled(x):
        calls.append(x)

    throttled(1)
    throttled(2)
    throttled(3)
    assert calls == [1, 3]


def test_seconds_attribute_is_not_enough(monkeypatch):
    # 1.5s: a period whose whole-second part alone would let the second call through.
    fake_clock(monkeypatch, [0, 1.4, 1.6])
    calls = []

    @throttle(timedelta(seconds=1, milliseconds=500))
    def throttled(x):
        calls.append(x)

    throttled(1)
    throttled(2)
    throttled(3)
    assert calls == [1, 3]


def test_period_longer_than_a_day(monkeypatch):
    fake_clock(monkeypatch, [0, 86_399.0, 86_400.5])
    calls = []

    @throttle(timedelta(days=1))
    def throttled(x):
        calls.append(x)

    throttled(1)
    throttled(2)
    throttled(3)
    assert calls == [1, 3]


def test_timedelta_and_equivalent_int_agree(monkeypatch):
    def run(period):
        fake_clock(monkeypatch, [0, 30, 90])
        calls = []

        @throttle(period)
        def throttled(x):
            calls.append(x)

        throttled(1)
        throttled(2)
        throttled(3)
        return calls

    assert run(timedelta(minutes=1)) == run(60) == [1, 3]
