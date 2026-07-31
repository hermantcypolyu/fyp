"""Visible smoke tests - the agent may run these via run_tests().

Deliberately weak. They confirm the package still imports and that a successful call
still returns, so the agent can catch itself breaking the module outright. They say
nothing about backoff, ceilings or jitter, because run_tests() must never hand over the
grading signal (AGENT_BUILD_SPEC 5.5).
"""

from src import ServiceClient
from src.retry import call_with_retry


def test_a_successful_call_returns_without_waiting():
    waits: list[float] = []
    assert call_with_retry(lambda: 42, attempts=3, delay=1.0, sleeper=waits.append) == 42
    assert waits == []


def test_the_example_client_fetches_a_document():
    client = ServiceClient(lambda path: f"body of {path}", sleeper=lambda _seconds: None)
    assert client.fetch("/a") == "body of /a"
