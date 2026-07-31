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


class TokenBucket:
    """Grant bursts of up to ``capacity`` units, refilled at ``refill_per_s`` per second.

    Where the fixed window resets in steps and is therefore wrong at the boundary, the
    bucket drips: the long-run rate is ``refill_per_s`` and ``capacity`` is only how much
    of that rate a quiet caller is allowed to save up and spend at once.

    The level is computed lazily. Rather than a background task topping every bucket up
    on a timer, each bucket remembers the clock reading at its last refill and settles up
    when someone asks a question - so an idle bucket costs nothing at all.
    """

    def __init__(self, capacity: int, refill_per_s: float, clock: Clock) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if refill_per_s < 0:
            raise ValueError("refill_per_s must be >= 0")
        self._capacity = float(capacity)
        self._refill_per_s = float(refill_per_s)
        self._clock = clock
        self._tokens = float(capacity)
        self._refilled_at = clock()

    @property
    def capacity(self) -> int:
        """The most tokens this bucket will hold, and the largest grantable request."""
        return int(self._capacity)

    @property
    def refill_per_s(self) -> float:
        """Tokens added per second of elapsed time."""
        return self._refill_per_s

    @property
    def tokens(self) -> float:
        """The level right now, settling up the pending refill first.

        Deliberately read-only and deliberately unrounded: callers use it for logging
        and for deciding how long to wait, and a rounded level would make the second of
        those wrong by up to half a token.
        """
        self._refill()
        return self._tokens

    def allow(self, n: int = 1) -> bool:
        """Spend ``n`` tokens if the bucket holds them, and say whether it did.

        All or nothing: a refused request leaves the level untouched, so a large request
        cannot drain the bucket on its way to failing.
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        if n > self._capacity:
            raise ValueError("n exceeds capacity; the request could never be granted")
        self._refill()
        if self._tokens < n:
            return False
        self._tokens -= n
        return True

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._refilled_at
        if elapsed <= 0:
            # A clock that has gone backwards - a rebooted peer, a restored snapshot -
            # must not hand out tokens when it catches up again, so keep the high-water
            # reading and charge nothing for the interval.
            return
        self._refilled_at = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_s)
