"""A very small HTTP client that does not own a socket.

The transport is injected, which is the whole point of the module: the client owns the
retry loop, the URL joining and the status policy, and something else owns the network.
A transport is any object with

    send(method, url, *, body, headers, timeout) -> Response

that raises :class:`TransportError` when it could not deliver the request at all.
:mod:`src.transports` has one that answers from a script, which is what the tests and the
local development server use.

Docstrings in this module follow the house style described in README.md: one ``:param``
line per parameter, in signature order, recording the default as ``(default ``x``)``
where there is one, and one ``:raises`` line per exception the call can actually produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlencode

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_TIMEOUT",
    "HTTPClient",
    "HTTPStatusError",
    "Response",
    "TransportError",
    "build_url",
]

#: Seconds a single attempt is given before the transport should give up.
DEFAULT_TIMEOUT = 5.0

#: How many times one request is handed to the transport before the client gives up.
DEFAULT_MAX_ATTEMPTS = 3


class TransportError(RuntimeError):
    """Raised when the transport could not deliver a request at all."""


class HTTPStatusError(RuntimeError):
    """Raised when the server answered with a status the client treats as a failure."""


@dataclass(frozen=True)
class Response:
    """One answer from a server: a status, its headers and its body as text."""

    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""

    @property
    def ok(self) -> bool:
        """True when the status is in the 2xx range."""
        return 200 <= self.status < 300


def build_url(base: str, path: str, query: dict[str, str] | None = None) -> str:
    joined = f"{base.rstrip('/')}/{path.lstrip('/')}" if base else path
    if not query:
        return joined
    return f"{joined}?{urlencode(sorted(query.items()))}"


class HTTPClient:
    """A client that retries, joins URLs and turns failing statuses into exceptions."""

    def __init__(
        self,
        transport: object,
        base_url: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        raise_for_status: bool = True,
    ) -> None:
        """Create a client that talks through ``transport``.

        :param transport: the object every request is handed to. It must expose
            ``send(method, url, *, body, headers, timeout)`` and raise
            :class:`TransportError` when it cannot deliver.
        :param url: prefix joined to the front of every path (default ``''``).
        :param timeout: seconds the transport is given for one attempt
            (default ``30``).
        :param retries: how many times one request is handed to the transport before
            the client gives up (default ``5``).
        :param raise_for_status: whether a failing status becomes an exception
            (default ``True``).
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self.transport = transport
        self.base_url = base_url
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.raise_for_status = raise_for_status

    def request(self, method: str, path: str, body: str | None = None, headers: dict[str, str] | None = None) -> Response:
        """Send one request and return the answer.

        :param method: the HTTP verb; it is upper-cased before it is sent.
        :param url: the path to ask for, joined to the client's prefix.
        :param headers: extra headers for this request alone (default ``{}``).
        :param payload: the request body, or ``None`` for a request without one
            (default ``None``).
        :raises ValueError: if the transport could not deliver the request, or the
            answer carried a status the client was told to treat as a failure.
        :returns: the :class:`Response` the transport produced.
        """
        url = build_url(self.base_url, path)
        verb = method.upper()
        last_error: Exception | None = None
        for _attempt in range(self.max_attempts):
            try:
                response = self.transport.send(
                    verb, url, body=body, headers=dict(headers or {}), timeout=self.timeout
                )
            except TransportError as exc:
                last_error = exc
                continue
            if self.raise_for_status and response.status >= 400:
                raise HTTPStatusError(f"{verb} {url} -> {response.status}")
            return response
        raise TransportError(f"{verb} {url} failed after {self.max_attempts} attempts") from last_error

    def get(self, path: str, headers: dict[str, str] | None = None) -> Response:
        """Ask for ``path`` without a body.

        :param path: the path to ask for, joined to the client's prefix.
        :param headers: extra headers for this request alone (default ``None``).
        :param body: ignored; a GET is always sent without one (default ``None``).
        :returns: the :class:`Response` the transport produced.
        """
        return self.request("GET", path, headers=headers)

    def post(self, path: str, body: str | None = None, headers: dict[str, str] | None = None) -> Response:
        """Send ``body`` to ``path``.

        :param path: the path to ask for, joined to the client's prefix.
        :param body: the request body (default ``None``).
        :param headers: extra headers for this request alone (default ``None``).
        :returns: the :class:`Response` the transport produced.
        """
        return self.request("POST", path, body=body, headers=headers)
