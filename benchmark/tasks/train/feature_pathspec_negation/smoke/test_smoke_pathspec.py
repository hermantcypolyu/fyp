"""Visible smoke tests - the agent may run these via run_tests().

Deliberately weak. They confirm the package still imports and that a plain exclusion
pattern still works end to end, so the agent can catch itself breaking the module
outright. They say nothing about ordering or negation, because run_tests() must never
hand over the grading signal (AGENT_BUILD_SPEC 5.5).
"""

from src import PathSpec, filter_paths


def test_a_plain_pattern_still_excludes():
    spec = PathSpec(["*.log"])
    assert spec.is_ignored("debug.log") is True
    assert spec.is_ignored("main.py") is False


def test_filtering_a_listing_keeps_the_rest_in_order():
    spec = PathSpec(["*.log"])
    listing = ["src/main.py", "debug.log", "README.md"]
    assert filter_paths(spec, listing) == ["src/main.py", "README.md"]
