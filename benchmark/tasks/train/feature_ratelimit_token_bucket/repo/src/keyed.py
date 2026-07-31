"""One limiter per caller, built on demand.

A single limiter in front of a shared service throttles everyone together, which is
almost never what an operator means by "rate limit". :class:`KeyedLimiter` keeps one
limiter per key and knows nothing about how that limiter decides.
"""

from __future__ import annotations

from typing import Callable, Hashable, Protocol


class Limiter(Protocol):
    """Anything that can be asked for ``n`` units and answer yes or no."""

    def allow(self, n: int = 1) -> bool: ...


class KeyedLimiter:
    """Give each key its own limiter, created by ``factory`` the first time it is seen."""

    def __init__(self, factory: Callable[[], Limiter]) -> None:
        self._factory = factory
        self._limiters: dict[Hashable, Limiter] = {}

    def allow(self, key: Hashable, n: int = 1) -> bool:
        """Charge ``n`` units against ``key``'s own limiter."""
        return self.for_key(key).allow(n)

    def for_key(self, key: Hashable) -> Limiter:
        """The limiter belonging to ``key``, created on first use."""
        if key not in self._limiters:
            self._limiters[key] = self._factory()
        return self._limiters[key]

    def forget(self, key: Hashable) -> None:
        """Drop ``key``'s limiter, so its next request is judged by a fresh one."""
        self._limiters.pop(key, None)

    def __len__(self) -> int:
        return len(self._limiters)
