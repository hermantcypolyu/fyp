"""A least-recently-used cache you can hold on to.

:func:`functools.lru_cache` is faster and better tested, but it is a decorator over one
function. We need an object: something a call site can be handed, sized per use, shared
between two functions, and inspected from the outside.

Recency is carried by the insertion order of the backing dict, which Python guarantees.
The first key is the least recently used and the last is the most recently used.
"""

from __future__ import annotations

from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

DEFAULT_CAPACITY = 128


class LRUCache(Generic[K, V]):
    """A mapping that forgets whichever entry it has gone longest without reading."""

    def __init__(self, capacity: int = DEFAULT_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        self._entries: dict[K, V] = {}

    @property
    def capacity(self) -> int:
        """The most entries this cache will hold."""
        return self._capacity

    def get(self, key: K, default: object = None) -> object:
        """Return the value stored under ``key``, marking it most recently used.

        Missing keys return ``default`` rather than raising: a cache read that misses
        is the normal case, not an error.
        """
        if key not in self._entries:
            return default
        value = self._entries.pop(key)
        self._entries[key] = value
        return value

    def put(self, key: K, value: V) -> None:
        """Store ``value`` under ``key``, dropping old entries to stay within capacity.

        Writing a key that is already present replaces its value and makes it the most
        recently used; it never grows the cache, so it never displaces anything.
        """
        if key in self._entries:
            del self._entries[key]
        self._entries[key] = value
        while len(self._entries) > self._capacity:
            oldest = next(iter(self._entries))
            del self._entries[oldest]

    def clear(self) -> None:
        """Drop every entry. Not an eviction: nothing was displaced by pressure."""
        self._entries.clear()

    def keys(self) -> list[K]:
        """Every key, from least to most recently used."""
        return list(self._entries)

    def __contains__(self, key: object) -> bool:
        """Membership without touching recency - a peek, not a read."""
        return key in self._entries

    def __len__(self) -> int:
        return len(self._entries)
