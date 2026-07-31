"""tinyretry - retrying transient failures without taking on a dependency."""

from .client import FlakyServiceError, ServiceClient
from .retry import DEFAULT_ATTEMPTS, DEFAULT_DELAY, Sleeper, call_with_retry, retry

__all__ = [
    "DEFAULT_ATTEMPTS",
    "DEFAULT_DELAY",
    "FlakyServiceError",
    "ServiceClient",
    "Sleeper",
    "call_with_retry",
    "retry",
]
