"""Memoising a pure function with an :class:`~src.cache.LRUCache` you can still reach.

The cache is attached to the wrapper as ``.cache``, so a caller can size it, empty it
between requests, or ask it how it is doing.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from .cache import DEFAULT_CAPACITY, LRUCache

T = TypeVar("T")

#: Sentinel distinguishing "not cached" from a cached value of None.
_MISS = object()


def memoise(capacity: int = DEFAULT_CAPACITY) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Cache a function's return value against its positional arguments.

    Keyword arguments are not supported: they would need a canonical ordering to key
    on, and every scheme for that is a bug waiting to happen.
    """

    def decorate(fn: Callable[..., T]) -> Callable[..., T]:
        cache: LRUCache[tuple[Any, ...], T] = LRUCache(capacity)

        @functools.wraps(fn)
        def wrapper(*args: Any) -> T:
            cached = cache.get(args, _MISS)
            if cached is not _MISS:
                return cached  # type: ignore[return-value]
            value = fn(*args)
            cache.put(args, value)
            return value

        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper

    return decorate
