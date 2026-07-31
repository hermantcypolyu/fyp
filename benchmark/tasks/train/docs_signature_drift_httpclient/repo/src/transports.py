"""Transports that answer from a script instead of from a socket.

Used by the test suite and by the local development server. A real transport lives in
the deployment package and is not imported here, which is why this repository has no
network code in it at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from .httpclient import Response, TransportError

__all__ = ["SentRequest", "StubTransport"]


@dataclass(frozen=True)
class SentRequest:
    """One call a transport received, kept so a test can look at it afterwards."""

    method: str
    url: str
    body: str | None
    headers: dict[str, str]
    timeout: float


class StubTransport:
    """Answers with canned responses, optionally refusing the first few attempts."""

    def __init__(self, responses: list[Response] | None = None, failures: int = 0) -> None:
        """Queue ``responses`` and refuse the first ``failures`` calls.

        :param responses: answers handed out in order; once they run out every call is
            answered with an empty ``200`` (default ``None``).
        :param failures: how many leading calls raise :class:`TransportError` before any
            answer is given (default ``0``).
        """
        self._responses = list(responses or [])
        self._failures = failures
        self.calls: list[SentRequest] = []

    def send(self, method: str, url: str, body: str | None = None, headers: dict[str, str] | None = None, timeout: float = 0.0) -> Response:
        """Record the call and answer it from the script.

        :param method: the HTTP verb the client chose.
        :param url: the fully joined URL the client built.
        :param body: the request body, if there was one (default ``None``).
        :param headers: the headers the client assembled (default ``None``).
        :param timeout: the per-attempt timeout the client passed (default ``0.0``).
        :raises TransportError: while the queued failure count has not run out.
        :returns: the next queued :class:`Response`, or an empty ``200``.
        """
        self.calls.append(SentRequest(method, url, body, dict(headers or {}), timeout))
        if self._failures > 0:
            self._failures -= 1
            raise TransportError(f"no route to {url}")
        if not self._responses:
            return Response(200)
        return self._responses.pop(0)
