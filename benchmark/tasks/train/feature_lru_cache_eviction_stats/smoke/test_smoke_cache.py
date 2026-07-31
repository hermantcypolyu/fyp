"""Visible smoke tests - the agent may run these via run_tests().

Deliberately weak. They confirm the package still imports and that storing, reading and
memoising still work, so the agent can catch itself breaking the module outright. They
say nothing about counters, because run_tests() must never hand over the grading signal
(AGENT_BUILD_SPEC 5.5).
"""

from src import LRUCache, memoise


def test_a_value_survives_a_round_trip():
    cache = LRUCache(capacity=3)
    cache.put("k", "v")
    assert cache.get("k") == "v"
    assert len(cache) == 1


def test_memoise_calls_the_function_once_per_distinct_argument():
    calls: list[int] = []

    @memoise(capacity=4)
    def double(n: int) -> int:
        calls.append(n)
        return n * 2

    assert [double(2), double(2), double(3)] == [4, 4, 6]
    assert calls == [2, 3]
