"""Progress is recoverable from rollouts recorded *before* the component existed.

The claim these tests protect is the one that decides whether Phase 3 has a measurable
premise at all. An earlier draft of `docs/reward_v2_design.md` said the per-turn test counts
were simply absent from the rollout logs, so component B was `None` everywhere and could not
be evaluated offline. That was wrong: `run_tests` writes `"N/M passed (smoke tests only; ...)"`
into `ToolResult.output`, the rollout JSONL stores the whole tool transcript, so the delta is
recoverable with a scrape and no environment change.

Having recovered it, the number turned out to be the interesting part — 95 rows ran the smoke
suite and none of them ever improved. `test_no_historical_rollout_ever_improved` pins that
finding to the data rather than to prose, so a future run that *does* show progress will make
this test fail loudly and prompt the design document to be updated with it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from rewards.v2 import progress_reward
from scripts.reward_distribution import progress_availability, smoke_counts_of, smoke_delta_of

RAW = Path("results/raw")


def _row(outputs: list[str]) -> dict[str, Any]:
    """A minimal rollout row shaped like the real ones: one entry per tool call."""
    return {"response": {"output": [{"output": out} for out in outputs]}}


# --- scraping the transcript -------------------------------------------------------------


def test_counts_are_read_in_call_order():
    row = _row(["0/2 passed (smoke tests only; these are not the grading tests)", "2/2 passed"])
    assert smoke_counts_of(row) == [0, 2]


def test_a_call_that_is_not_run_tests_contributes_nothing():
    """The transcript is every tool call, not just the test runs."""
    row = _row(["file written", "6/11 passed", '{"ok": true}'])
    assert smoke_counts_of(row) == [6]


def test_a_row_that_never_ran_tests_has_no_measurement():
    assert smoke_counts_of(_row(["cat: no such file"])) == []
    assert smoke_delta_of(_row(["cat: no such file"])) is None
    assert progress_reward(None) is None


# --- the delta, against the same rules as `rewards.v2.smoke_delta` ------------------------


def test_the_baseline_is_the_first_reading_and_the_result_is_the_best():
    row = _row(["6/11 passed", "8/11 passed", "9/11 passed"])
    assert smoke_delta_of(row) == 3


def test_improving_then_breaking_it_still_earns_the_improvement():
    row = _row(["6/11 passed", "10/11 passed", "6/11 passed"])
    assert smoke_delta_of(row) == 4


def test_ending_below_the_baseline_is_zero_not_negative():
    row = _row(["6/11 passed", "4/11 passed"])
    assert smoke_delta_of(row) == 0


def test_the_single_reading_case_is_always_zero():
    """One test call can show no delta: progress needs at least two observations."""
    assert smoke_delta_of(_row(["2/2 passed"])) == 0


# --- the finding, pinned to the data -----------------------------------------------------


@pytest.mark.skipif(not RAW.is_dir(), reason="no results/raw in this checkout")
def test_progress_is_measurable_on_the_historical_rollouts():
    """If this fails the scrape has broken, not the model."""
    from scripts.reward_distribution import load_rollouts

    rows = load_rollouts(RAW)
    availability = progress_availability(rows)
    assert availability["rows"] > 0
    assert availability["rows_with_a_test_run"] > 0, "no rollout on disk exposes a test count"


@pytest.mark.skipif(not RAW.is_dir(), reason="no results/raw in this checkout")
def test_no_historical_rollout_ever_improved():
    """The measured basis for calling B unexercised rather than merely unshipped.

    If this starts failing, a run has produced progress the reward can finally price. That is
    good news, but it means `docs/reward_v2_design.md` §6 and the claim that "the model never
    performs the edit-then-retest loop" are both out of date and must be rewritten with the
    new numbers - which is why the test is a tripwire and not a tautology.
    """
    from scripts.reward_distribution import load_rollouts

    availability = progress_availability(load_rollouts(RAW))
    assert availability["rows_with_positive_progress"] == 0, (
        "a rollout now shows positive progress; update docs/reward_v2_design.md section 6"
    )
    assert availability["max_observed_delta"] == 0


@pytest.mark.skipif(not RAW.is_dir(), reason="no results/raw in this checkout")
def test_the_measured_inputs_to_component_b_are_recorded_in_the_report():
    report = Path("results/analysis/reward_v2_offline.json")
    if not report.is_file():
        pytest.skip("run scripts/reward_distribution.py --json results/analysis/reward_v2_offline.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    availability = data["progress_availability"]
    assert availability["rows"] == data["rows"]
    assert availability["rows_with_a_test_run"] <= availability["rows"]
