"""Retrying calls that fail transiently.

The sleep function is a parameter rather than a module-level import, so a policy can be
exercised in a test without spending real time: pass a recorder in place of
``time.sleep`` and assert on the sequence of waits it was handed.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")

#: How many times a call is attempted in total, not how many times it is re-tried.
DEFAULT_ATTEMPTS = 3
#: Seconds between attempts.
DEFAULT_DELAY = 0.5

#: Anything that can stand in for ``time.sleep``.
Sleeper = Callable[[float], None]


def call_with_retry(
    fn: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    delay: float = DEFAULT_DELAY,
    on: tuple[type[BaseException], ...] = (Exception,),
    sleeper: Sleeper = time.sleep,
) -> T:
    """Call ``fn`` until it succeeds, waiting between attempts.

    ``fn`` takes no arguments; bind them with :func:`functools.partial` or a lambda.
    Only the exception types listed in ``on`` are retried - anything else propagates
    straight away, because a ``TypeError`` from a bad call site will not fix itself on
    the third go. When the budget runs out the exception from the final attempt is
    re-raised as-is, so callers keep the original traceback.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    if delay < 0:
        raise ValueError("delay must be >= 0")

    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except on:
            if attempt == attempts:
                raise
            sleeper(_wait_after(attempt, delay))
    raise AssertionError("unreachable: the loop either returns or raises")


def _wait_after(attempt: int, delay: float) -> float:
    """Seconds to wait after the ``attempt``-th failure. Attempts are 1-based."""
    return delay


def retry(
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    delay: float = DEFAULT_DELAY,
    on: tuple[type[BaseException], ...] = (Exception,),
    sleeper: Sleeper = time.sleep,
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
            )

        return wrapper

    return decorate
