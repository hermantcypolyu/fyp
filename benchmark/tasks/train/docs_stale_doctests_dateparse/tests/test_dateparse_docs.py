"""Hidden verification suite for docs_stale_doctests_dateparse.

Never mounted into the sandbox. Fourteen independent assertions, no parametrisation, so
that a partly-corrected file earns partial credit (AGENT_BUILD_SPEC 5.4 rule 5).

The suite is deliberately two-sided, because the documentation and the code live in the
same file and the agent has to be able to edit that file. The first seven tests pin the
module's *real* behaviour: an agent that "fixes" a lying example by bending the code
until the stale text becomes true fails those, which is the point. The remaining seven
run the documented examples themselves - the module docstring, five function docstrings,
and the whole module in one pass - so each corrected docstring is worth something on its
own.

Nothing here grades prose. Every assertion is a property of the module as imported.
"""

from __future__ import annotations

import doctest
import inspect
from datetime import date
from pathlib import Path

import pytest

import src.dateparse as dateparse
from src.dateparse import (
    DateParseError,
    add_business_days,
    days_between,
    format_human,
    month_bounds,
    parse_iso_date,
    weekday_name,
)

DOCUMENTED = (parse_iso_date, format_human, days_between, add_business_days, month_bounds, weekday_name)


def _run_examples(obj: object, *, recurse: bool = False) -> str:
    """Run the examples attached to ``obj``; return "" when they all hold, else a report."""
    finder = doctest.DocTestFinder(recurse=recurse, exclude_empty=False)
    runner = doctest.DocTestRunner(verbose=False)
    chunks: list[str] = []
    for test in finder.find(obj, module=dateparse, globs=dict(vars(dateparse))):
        runner.run(test, out=chunks.append)
    return "" if runner.failures == 0 else "".join(chunks)


def _example_counts() -> dict[str, int]:
    finder = doctest.DocTestFinder(recurse=True, exclude_empty=False)
    return {t.name.rsplit(".", 1)[-1]: len(t.examples) for t in finder.find(dateparse, module=dateparse)}


# --- what the module really does: unchanged by the fix, and must stay that way --------


def test_parse_iso_date_returns_a_real_date_object():
    parsed = parse_iso_date("2024-03-05")
    assert isinstance(parsed, date)
    assert parsed == date(2024, 3, 5)
    assert parse_iso_date(" 2024-03-05 ") == date(2024, 3, 5)


def test_an_unreadable_string_raises_date_parse_error():
    with pytest.raises(DateParseError):
        parse_iso_date("05/03/2024")
    with pytest.raises(DateParseError):
        parse_iso_date("2024-02-30")


def test_format_human_writes_the_day_before_the_month():
    assert format_human(date(2024, 3, 5)) == "5 March 2024"
    assert format_human(date(2024, 12, 25)) == "25 December 2024"


def test_days_between_returns_a_signed_whole_number():
    gap = days_between(date(2024, 3, 1), date(2024, 3, 10))
    assert isinstance(gap, int)
    assert gap == 9
    assert days_between(date(2024, 3, 10), date(2024, 3, 1)) == -9


def test_add_business_days_steps_over_the_weekend():
    assert add_business_days(date(2024, 3, 7), 2) == date(2024, 3, 11)
    assert add_business_days(date(2024, 3, 8), 1) == date(2024, 3, 11)
    assert add_business_days(date(2024, 3, 7), 0) == date(2024, 3, 7)


def test_month_bounds_returns_a_pair_and_knows_about_leap_years():
    assert isinstance(month_bounds(2024, 2), tuple)
    assert month_bounds(2024, 2) == (date(2024, 2, 1), date(2024, 2, 29))
    assert month_bounds(2023, 2) == (date(2023, 2, 1), date(2023, 2, 28))
    assert month_bounds(2024, 12) == (date(2024, 12, 1), date(2024, 12, 31))


def test_the_examples_are_still_there_and_none_of_them_are_muted():
    source = Path(inspect.getsourcefile(dateparse)).read_text(encoding="utf-8")
    assert "# doctest:" not in source, "an example was silenced with a doctest directive"
    counts = _example_counts()
    assert counts.get("dateparse", 0) >= 2, "the module docstring lost its examples"
    for fn in DOCUMENTED:
        assert counts.get(fn.__name__, 0) >= 2, f"{fn.__name__} no longer documents two examples"


# --- what the module claims it does ---------------------------------------------------


def test_the_module_docstring_examples_run():
    assert _run_examples(dateparse) == ""


def test_the_parse_iso_date_examples_run():
    assert _run_examples(parse_iso_date) == ""


def test_the_format_human_examples_run():
    assert _run_examples(format_human) == ""


def test_the_days_between_examples_run():
    assert _run_examples(days_between) == ""


def test_the_add_business_days_examples_run():
    assert _run_examples(add_business_days) == ""


def test_the_month_bounds_examples_run():
    assert _run_examples(month_bounds) == ""


def test_every_example_in_the_module_runs():
    assert _run_examples(dateparse, recurse=True) == ""
