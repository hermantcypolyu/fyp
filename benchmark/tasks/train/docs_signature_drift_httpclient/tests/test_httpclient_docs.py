"""Hidden verification suite for docs_signature_drift_httpclient.

Never mounted into the sandbox. Fifteen independent assertions, no parametrisation, so
that a partly-corrected file earns partial credit (AGENT_BUILD_SPEC 5.4 rule 5).

Docstrings live inside the module, so the agent has to be able to write the module - and
that leaves an obvious cheat: rename ``base_url`` to ``url`` and change ``timeout`` to
30, and the stale documentation becomes true. The first eight tests exist to close it.
They pin the client's real behaviour, the real parameter names in their real order, and
the real defaults, so an agent that edits the code to match the prose fails more tests
than it fixes. The last seven read the docstrings and compare them with the module as
imported: names against `inspect.signature`, documented defaults against real defaults,
documented exception types against the types two failing calls actually raise.

Nothing here grades prose. Every assertion is a property of the module as imported.
"""

from __future__ import annotations

import ast
import inspect
import re

import pytest

import src.httpclient as httpclient
from src.httpclient import DEFAULT_MAX_ATTEMPTS, DEFAULT_TIMEOUT, HTTPClient, Response, build_url
from src.transports import StubTransport

PARAM_RE = re.compile(r"^\s*:param\s+([A-Za-z_][A-Za-z0-9_]*)\s*:(.*)$")
FIELD_RE = re.compile(r"^\s*:(param|raises|returns|rtype)\b")
RAISES_RE = re.compile(r"^\s*:raises\s+([A-Za-z_][A-Za-z0-9_.]*)\s*:")
DEFAULT_RE = re.compile(r"\(default\s+``(.+?)``\)")

MISSING = object()


def _param_blocks(doc: str | None) -> dict[str, str]:
    """Map each documented parameter to its description, in the order documented."""
    blocks: dict[str, str] = {}
    current: str | None = None
    for line in (doc or "").splitlines():
        match = PARAM_RE.match(line)
        if match:
            current = match.group(1)
            blocks[current] = match.group(2).strip()
        elif FIELD_RE.match(line):
            current = None
        elif current is not None:
            blocks[current] = f"{blocks[current]} {line.strip()}".strip()
    return blocks


def _real_params(fn: object) -> list[str]:
    return [
        name
        for name, p in inspect.signature(fn).parameters.items()
        if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def _real_defaults(fn: object) -> dict[str, object]:
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if name != "self" and p.default is not inspect.Parameter.empty
    }


def _documented_default(block: str) -> object:
    match = DEFAULT_RE.search(block)
    if match is None:
        return MISSING
    try:
        return ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return MISSING


def _documented_raises(doc: str | None) -> set[str]:
    return {m.group(1) for line in (doc or "").splitlines() if (m := RAISES_RE.match(line))}


def _public_callables() -> list[tuple[str, object]]:
    found: list[tuple[str, object]] = []
    for name, obj in vars(httpclient).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != httpclient.__name__:
            continue
        if inspect.isclass(obj):
            found.append((name, obj))
            found += [
                (f"{name}.{attr}", member)
                for attr, member in vars(obj).items()
                if not attr.startswith("_") and inspect.isfunction(member)
            ]
        elif inspect.isfunction(obj):
            found.append((name, obj))
    return found


def _raised_by(call) -> str:
    try:
        call()
    except Exception as exc:  # noqa: BLE001 - observing the type is the whole point
        return type(exc).__name__
    raise AssertionError("expected that call to raise, and it did not")


# --- what the client really does, and must go on doing --------------------------------


def test_a_get_reaches_the_transport_with_the_verb_and_url_the_caller_asked_for():
    transport = StubTransport([Response(200, {}, "pong")])
    client = HTTPClient(transport, base_url="https://api.example.com")
    assert client.get("/ping").body == "pong"
    sent = transport.calls[0]
    assert (sent.method, sent.url, sent.body) == ("GET", "https://api.example.com/ping", None)


def test_build_url_joins_the_prefix_and_sorts_the_query():
    assert build_url("https://api.example.com/", "/v1/things") == "https://api.example.com/v1/things"
    assert build_url("", "/v1/things") == "/v1/things"
    assert build_url("https://x.test", "y", {"b": "2", "a": "1"}) == "https://x.test/y?a=1&b=2"


def test_a_failing_status_becomes_an_http_status_error():
    client = HTTPClient(StubTransport([Response(404)]))
    with pytest.raises(httpclient.HTTPStatusError):
        client.get("/missing")


def test_a_client_told_not_to_raise_hands_the_failing_response_back():
    client = HTTPClient(StubTransport([Response(500, {}, "boom")]), raise_for_status=False)
    response = client.get("/unwell")
    assert response.status == 500
    assert response.ok is False


def test_a_refused_attempt_is_tried_again():
    transport = StubTransport([Response(200, {}, "late")], failures=2)
    assert HTTPClient(transport).post("/thing", body="x").body == "late"
    assert len(transport.calls) == 3


def test_a_transport_that_never_answers_ends_in_a_transport_error():
    transport = StubTransport(failures=9)
    with pytest.raises(httpclient.TransportError):
        HTTPClient(transport, max_attempts=2).get("/thing")
    assert len(transport.calls) == 2


def test_the_public_parameter_names_are_unchanged():
    assert _real_params(HTTPClient.__init__) == [
        "transport",
        "base_url",
        "timeout",
        "max_attempts",
        "raise_for_status",
    ]
    assert _real_params(HTTPClient.request) == ["method", "path", "body", "headers"]
    assert _real_params(HTTPClient.get) == ["path", "headers"]
    assert _real_params(HTTPClient.post) == ["path", "body", "headers"]
    assert _real_params(build_url) == ["base", "path", "query"]


def test_the_real_defaults_are_unchanged_and_reach_the_transport():
    assert _real_defaults(HTTPClient.__init__) == {
        "base_url": "",
        "timeout": 5.0,
        "max_attempts": 3,
        "raise_for_status": True,
    }
    assert (DEFAULT_TIMEOUT, DEFAULT_MAX_ATTEMPTS) == (5.0, 3)
    transport = StubTransport()
    HTTPClient(transport).get("/thing")
    assert transport.calls[0].timeout == 5.0


# --- what the docstrings claim --------------------------------------------------------


def test_every_public_callable_carries_a_docstring():
    undocumented = [name for name, obj in _public_callables() if not (obj.__doc__ or "").strip()]
    assert undocumented == []


def test_the_constructor_documents_the_parameters_it_takes():
    assert list(_param_blocks(HTTPClient.__init__.__doc__)) == _real_params(HTTPClient.__init__)


def test_request_documents_the_parameters_it_takes():
    assert list(_param_blocks(HTTPClient.request.__doc__)) == _real_params(HTTPClient.request)


def test_get_documents_the_parameters_it_takes():
    assert list(_param_blocks(HTTPClient.get.__doc__)) == _real_params(HTTPClient.get)


def test_the_documented_constructor_defaults_are_the_real_ones():
    blocks = _param_blocks(HTTPClient.__init__.__doc__)
    for name, default in _real_defaults(HTTPClient.__init__).items():
        assert name in blocks, f"{name} has a default and no :param line"
        assert _documented_default(blocks[name]) == default, f"the documented default for {name} is wrong"


def test_the_documented_request_defaults_are_the_real_ones():
    blocks = _param_blocks(HTTPClient.request.__doc__)
    for name, default in _real_defaults(HTTPClient.request).items():
        assert name in blocks, f"{name} has a default and no :param line"
        assert _documented_default(blocks[name]) == default, f"the documented default for {name} is wrong"


def test_request_documents_the_exceptions_it_can_raise():
    documented = _documented_raises(HTTPClient.request.__doc__)
    for name in sorted(documented):
        assert hasattr(httpclient, name), f":raises {name}: names a type this module does not define"
    observed = {
        _raised_by(lambda: HTTPClient(StubTransport(failures=9), max_attempts=2).request("GET", "/thing")),
        _raised_by(lambda: HTTPClient(StubTransport([Response(503)])).request("GET", "/thing")),
    }
    assert documented == observed
