"""Hidden verification for `docs_interval_docstring_drift`.

The code in `intervals/` is correct and stays correct: every behaviour test below passes
against the untouched repo, and they are here so that an agent which "fixes" a failing
example by changing the implementation is caught immediately.

What has drifted is the documentation. Each public function carries a worked example in
doctest form plus `:param:`/`:returns:` fields, and the examples were last updated before
the unit handling was reworked. The drifts are independent - a different function each
time, plus three prose claims that contradict the signature or the return value - so an
agent that corrects some of them scores strictly between the floor and 1.0.

Each doctest test also asserts that the function *still has* an example, which closes the
cheap answer of deleting the examples that fail.
"""

import doctest
import inspect
from typing import Tuple

import pytest

from intervals import core

PUBLIC = [
    "parse_duration",
    "format_duration",
    "format_long",
    "split_units",
    "total_seconds",
    "clamp_duration",
]


def check_doctest(name: str) -> None:
    """Run just this function's examples, and insist it still has some."""
    func = getattr(core, name)
    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    captured: list[str] = []
    examples = 0
    for test in finder.find(func, name, globs=dict(vars(core))):
        if not test.examples:
            continue
        examples += len(test.examples)
        runner.run(test, out=captured.append)
    assert examples, f"{name}() no longer documents a worked example; update it, do not delete it"
    assert runner.failures == 0, (
        f"{name}() documents an example that disagrees with the code:\n{''.join(captured)}"
    )


def doc_of(name: str) -> str:
    doc = inspect.getdoc(getattr(core, name))
    assert doc, f"{name}() has lost its docstring"
    return doc


# --- the documented examples must match what the code does ------------------------------


def test_parse_duration_example_is_accurate():
    check_doctest("parse_duration")


def test_format_duration_example_is_accurate():
    check_doctest("format_duration")


def test_format_long_example_is_accurate():
    check_doctest("format_long")


def test_split_units_example_is_accurate():
    check_doctest("split_units")


def test_total_seconds_example_is_accurate():
    check_doctest("total_seconds")


def test_clamp_duration_example_is_accurate():
    check_doctest("clamp_duration")


# --- documented prose must match the signature and the return value ---------------------


def test_format_duration_documents_its_real_keyword():
    doc = doc_of("format_duration")
    declared = set(inspect.signature(core.format_duration).parameters)
    assert "precision" in declared
    assert ":param precision:" in doc, "the keyword argument is documented under a name it does not have"
    assert ":param digits:" not in doc


def test_split_units_documents_a_tuple_result():
    doc = doc_of("split_units").lower()
    assert "tuple" in doc, "split_units returns a tuple, but its :returns: does not say so"
    assert "0d 2h 30m 0s" not in doc, "split_units does not return a formatted string"


def test_clamp_duration_documents_clamping_not_none():
    doc = doc_of("clamp_duration").lower()
    assert "none" not in doc, "clamp_duration never returns None; it returns the nearest bound"


# --- the implementation, which must not be edited to make the docs true -----------------


def test_parse_duration_behaviour():
    assert core.parse_duration("2h30m") == 9000
    assert core.parse_duration("1d 12h") == 129600
    assert core.parse_duration("nothing") is None


def test_format_duration_behaviour():
    assert core.format_duration(9000) == "2.5h"
    assert core.format_duration(90, precision=0) == "2m"


def test_format_long_behaviour():
    assert core.format_long(9000) == "2.5 hours"


def test_split_units_behaviour():
    result = core.split_units(9000)
    assert isinstance(result, tuple)
    assert result == (0, 2, 30, 0)


def test_total_seconds_behaviour():
    assert core.total_seconds((0, 2, 30, 0)) == 9000


def test_clamp_duration_behaviour():
    assert core.clamp_duration(100, 0, 60) == 60
    assert core.clamp_duration(-5, 0, 60) == 0
    assert core.clamp_duration(30, 0, 60) == 30


def test_every_public_function_is_still_documented():
    for name in PUBLIC:
        assert doc_of(name).strip(), f"{name}() must keep a docstring"


def test_the_span_helpers_are_unaffected():
    from intervals import ranges

    assert ranges.length((0, 9000)) == 9000
    assert ranges.overlaps((0, 100), (50, 150))
    assert not ranges.overlaps((0, 100), (100, 150))
    assert ranges.merge([(0, 100), (50, 150), (200, 250)]) == [(0, 150), (200, 250)]
    assert ranges.total_length([(0, 100), (50, 150)]) == 150
    assert ranges.as_parts((0, 9000)) == (0, 2, 30, 0)
