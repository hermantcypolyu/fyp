"""Visible smoke checks - this is what run_tests() shows the agent.

Deliberately weak: they confirm the package still imports and that a request still gets
through the client to the transport and back. They say nothing about the docstrings,
because run_tests() must never hand over the grading signal (AGENT_BUILD_SPEC 5.5).
"""

from src import HTTPClient, Response, StubTransport


def test_a_response_comes_back_through_the_client():
    client = HTTPClient(StubTransport([Response(200, {}, "pong")]))
    assert client.get("/ping").body == "pong"


def test_the_prefix_ends_up_in_front_of_the_path():
    transport = StubTransport()
    HTTPClient(transport, base_url="https://api.example.com").get("/ping")
    assert transport.calls[0].url == "https://api.example.com/ping"
