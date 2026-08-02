"""Minimal stand-ins for the call-state objects a retry strategy is handed.

Upstream these live in ``tenacity/__init__.py`` next to the ``Retrying`` engine, which is
far too large to carry here. Only the attributes the strategies in :mod:`tenacity.retry`
actually read are reproduced: the outcome future and the attempt counter.
"""

import concurrent.futures
import typing


class Future(concurrent.futures.Future):
    """The outcome of one attempted call to the target function."""

    def __init__(self, attempt_number: int) -> None:
        super().__init__()
        self.attempt_number = attempt_number

    @property
    def failed(self) -> bool:
        """Whether the attempt ended in an exception rather than a value."""
        return self.exception() is not None

    @classmethod
    def construct(
        cls, attempt_number: int, value: typing.Any, has_exception: bool
    ) -> "Future":
        fut = cls(attempt_number)
        if has_exception:
            fut.set_exception(value)
        else:
            fut.set_result(value)
        return fut


class RetryCallState:
    """State of a retried call, as seen by a retry strategy."""

    def __init__(
        self, attempt_number: int = 1, outcome: typing.Optional[Future] = None
    ) -> None:
        self.attempt_number = attempt_number
        self.outcome = outcome

    def set_result(self, value: typing.Any) -> None:
        self.outcome = Future.construct(self.attempt_number, value, False)

    def set_exception(self, exc: BaseException) -> None:
        self.outcome = Future.construct(self.attempt_number, exc, True)
