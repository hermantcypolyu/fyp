"""A worked example of the retry helpers: a client over an unreliable transport.

The transport is injected, so this module has no network dependency of its own. In
production it would be a session object; in tests it is a function.
"""

from __future__ import annotations

import time
from typing import Callable

from .retry import DEFAULT_ATTEMPTS, DEFAULT_DELAY, Sleeper, call_with_retry


class FlakyServiceError(RuntimeError):
    """The upstream service failed in a way that is worth trying again."""


class ServiceClient:
    """Fetches documents, retrying only the failures the transport calls transient."""

    def __init__(
        self,
        transport: Callable[[str], str],
        *,
        attempts: int = DEFAULT_ATTEMPTS,
        delay: float = DEFAULT_DELAY,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        self._transport = transport
        self._attempts = attempts
        self._delay = delay
        self._sleeper = sleeper

    def fetch(self, path: str) -> str:
        """Return the body at ``path``, retrying :class:`FlakyServiceError`."""
        return call_with_retry(
            lambda: self._transport(path),
            attempts=self._attempts,
            delay=self._delay,
            on=(FlakyServiceError,),
            sleeper=self._sleeper,
        )
