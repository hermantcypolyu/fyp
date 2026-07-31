"""Request limiters that are told the time rather than asking for it.

Every limiter here takes a ``clock``: a zero-argument callable returning a float number
of seconds. Production passes :func:`time.monotonic`; tests pass something they move by
hand. Nothing in this module imports :mod:`time`, so a limiter can be driven through a
week of traffic in a millisecond and no test ever has to sleep.
"""

from __future__ import annotations

from typing import Callable

#: A source of seconds. Only the differences between two readings are ever used, so the
#: origin is arbitrary as long as it is stable for the life of one limiter.
Clock = Callable[[], float]


class FixedWindowLimiter:
    """Grant at most ``limit`` units per fixed window of ``window_s`` seconds.

    The cheapest useful limiter, and the one everybody writes first. Its known weakness
    is the boundary: a caller can spend a whole allowance at the end of one window and a
    whole allowance at the start of the next, so twice ``limit`` gets through in an
    instant. It also cannot express "slow on average, bursty in the moment".
    """

    def __init__(self, limit: int, window_s: float, clock: Clock) -> None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if window_s <= 0:
            raise ValueError("window_s must be > 0")
        self._limit = limit
        self._window_s = window_s
        self._clock = clock
        self._window_start = clock()
        self._used = 0

    @property
    def limit(self) -> int:
        """Units granted per window."""
        return self._limit

    @property
    def remaining(self) -> int:
        """Units still available now, rolling the window over first if it has expired."""
        self._roll()
        return self._limit - self._used

    def allow(self, n: int = 1) -> bool:
        """Spend ``n`` units if this window has room for them, and say whether it did.

        A refused request spends nothing. ``n`` outside ``1 .. limit`` is a mistake in
        the caller rather than backpressure, so it raises instead of returning ``False``.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if n > self._limit:
            raise ValueError("n exceeds limit; the request could never be granted")
        self._roll()
        if self._used + n > self._limit:
            return False
        self._used += n
        return True

    def _roll(self) -> None:
        now = self._clock()
        if now - self._window_start >= self._window_s:
            self._window_start = now
            self._used = 0
