"""Retrying calls that fail transiently.

The sleep function is a parameter rather than a module-level import, so a policy can be
exercised in a test without spending real time: pass a recorder in place of
``time.sleep`` and assert on the sequence of waits it was handed. The random source
behind ``jitter`` is injectable for exactly the same reason.
"""

from __future__ import annotations

import functools
import random
import time
from typing import Any, Callable, Protocol, TypeVar

T = TypeVar("T")

#: How many times a call is attempted in total, not how many times it is re-tried.
DEFAULT_ATTEMPTS = 3
#: Seconds between attempts.
DEFAULT_DELAY = 0.5
#: Multiplier applied once per elapsed attempt. 1.0 is a flat delay.
DEFAULT_BACKOFF = 1.0

#: Anything that can stand in for ``time.sleep``.
Sleeper = Callable[[float], None]


class RandomSource(Protocol):
    """The one method this module needs from :class:`random.Random`."""

    def uniform(self, a: float, b: float) -> float: ...


_DEFAULT_RNG = random.Random()


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    delay: float = DEFAULT_DELAY,
    on: tuple[type[BaseException], ...] = (Exception,),
    sleeper: Sleeper = time.sleep,
    backoff: float = DEFAULT_BACKOFF,
    max_delay: float | None = None,
    jitter: bool = False,
    rng: RandomSource | None = None,
) -> T:
    """Call ``fn`` until it succeeds, waiting between attempts.

    ``fn`` takes no arguments; bind them with :func:`functools.partial` or a lambda.
    Only the exception types listed in ``on`` are retried - anything else propagates
    straight away, because a ``TypeError`` from a bad call site will not fix itself on
    the third go. When the budget runs out the exception from the final attempt is
    re-raised as-is, so callers keep the original traceback.

    The wait after the *n*-th failure grows as ``delay * backoff ** (n - 1)``, is
    clamped to ``max_delay`` when that is not ``None``, and is then replaced by a draw
    from ``[0, wait]`` when ``jitter`` is set. The defaults - ``backoff=1.0``,
    ``max_delay=None``, ``jitter=False`` - are a flat ``delay`` every time.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if delay < 0:
        raise ValueError("delay must be >= 0")
    if backoff < 1.0:
        raise ValueError("backoff must be >= 1.0")
    if max_delay is not None and max_delay < 0:
        raise ValueError("max_delay must be >= 0")

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except on:
            if attempt == attempts:
                raise
            sleeper(_wait_after(attempt, delay, backoff, max_delay, jitter, rng))
    raise AssertionError("unreachable: the loop either returns or raises")


def _wait_after(
    attempt: int,
    delay: float,
    backoff: float = DEFAULT_BACKOFF,
    max_delay: float | None = None,
    jitter: bool = False,
    rng: RandomSource | None = None,
) -> float:
    """Seconds to wait after the ``attempt``-th failure. Attempts are 1-based."""
    wait = delay * backoff ** (attempt - 1)
    if max_delay is not None:
        wait = min(wait, max_delay)
    if jitter:
        wait = (rng if rng is not None else _DEFAULT_RNG).uniform(0.0, wait)
    return wait


def retry(
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    delay: float = DEFAULT_DELAY,
    on: tuple[type[BaseException], ...] = (Exception,),
    sleeper: Sleeper = time.sleep,
    backoff: float = DEFAULT_BACKOFF,
    max_delay: float | None = None,
    jitter: bool = False,
    rng: RandomSource | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form of :func:`call_with_retry`, taking the same options.

    >>> waits = []
    >>> @retry(attempts=2, delay=0.1, sleeper=waits.append)
    ... def ping() -> str:
    ...     return "pong"
    >>> ping()
    'pong'
    """

    def decorate(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return call_with_retry(
                functools.partial(fn, *args, **kwargs),
                attempts=attempts,
                delay=delay,
                on=on,
                sleeper=sleeper,
                backoff=backoff,
                max_delay=max_delay,
                jitter=jitter,
                rng=rng,
            )

        return wrapper

    return decorate
