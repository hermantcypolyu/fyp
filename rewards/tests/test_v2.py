"""Unit tests for `reward_v2`: correctness plus observed progress.

The test that matters most is `test_progress_cannot_buy_back_correctness`. reward_v2 is a
weighted sum, so every weight is a trade the agent can discover: give progress enough weight
and the cheapest way to raise reward becomes "learn to run the smoke tests", not "learn to
fix the bug". The weights are asserted against that, not just against their values.

The second tripwire is `test_a_gated_run_stays_at_the_bottom_of_the_scale`. Gates are the
only thing standing between the reward and the write-to-tests/ hack (ADR 0006), and a
component added to one branch of `compose_v2` and forgotten in the other would let a gated
run collect its progress term. That is the failure this file exists to catch.
"""

from __future__ import annotations

import statistics

import pytest

from rewards.compose import GATE_REWARD, REWARD_MIN, REWARD_SOLVED
from rewards.v2 import (
    CORRECTNESS_WEIGHT,
    MAX_SMOKE_DELTA,
    PROGRESS_WEIGHT,
    SHIPPED_COMPONENTS,
    WEIGHTS,
    compose_v2,
    progress_reward,
    smoke_delta,
)

# --- the weights, and the trade they encode ---------------------------------------------


def test_correctness_outweighs_progress_by_a_wide_margin():
    assert CORRECTNESS_WEIGHT > PROGRESS_WEIGHT
    assert CORRECTNESS_WEIGHT == pytest.approx(0.60)
    assert PROGRESS_WEIGHT == pytest.approx(0.20)


def test_a_solved_task_beats_a_perfect_progress_never_solved_task():
    """The trade must not be close enough to invert by accident."""
    solved_no_progress = compose_v2(14, 14, 6, progress=0.0)
    perfect_progress_no_solve = compose_v2(6, 14, 6, progress=1.0)
    assert solved_no_progress.final > perfect_progress_no_solve.final


def test_only_the_shipped_components_carry_weight():
    assert set(WEIGHTS) == {"correctness", "progress"}
    assert SHIPPED_COMPONENTS == ("correctness", "progress")
    # C and D are absent rather than weighted zero: "not measured" is not "scored zero".
    assert "validity" not in WEIGHTS
    assert "tool" not in WEIGHTS


def test_the_shipped_weights_do_not_sum_to_the_whole_scale():
    """Together they reach 0.80, and the remaining 0.20 is reserved, not redistributed.

    If they summed to 1.0 a solved rollout with unmeasured progress would score 1.0, and
    the later addition of C and D would silently deflate every historical comparison.
    """
    assert sum(WEIGHTS.values()) == pytest.approx(0.80)


# --- gates ------------------------------------------------------------------------------


@pytest.mark.parametrize("gate", ["tests_modified", "no_submit", "timeout", "syntax_broken"])
def test_a_gated_run_stays_at_the_bottom_of_the_scale(gate):
    """No combination of correctness and progress can lift a gated run."""
    best_possible = compose_v2(14, 14, 6, gates=[gate], progress=1.0)
    assert best_possible.final == GATE_REWARD
    assert best_possible.final == REWARD_MIN


def test_a_gated_run_still_reports_what_it_would_have_scored():
    """The components are diagnostics; only `final` is zeroed.

    A post-mortem needs to see that a gated run had in fact solved the task before the gate
    discarded it - that is the difference between "the model is bad" and "a guard fired".
    """
    gated = compose_v2(14, 14, 6, gates=["tests_modified"], progress=0.5)
    assert gated.final == GATE_REWARD
    assert gated.correctness == pytest.approx(REWARD_SOLVED)
    assert gated.progress == pytest.approx(0.5)
    assert gated.gates == ("tests_modified",)


def test_gate_names_are_deduplicated_and_ordered():
    gated = compose_v2(6, 14, 6, gates=["no_submit", "no_submit", "timeout"])
    assert gated.gates == ("no_submit", "timeout")


# --- progress: the mapping ---------------------------------------------------------------


def test_no_measurable_progress_is_none_not_zero():
    """A task with no smoke suite must not be scored as though it made zero progress."""
    assert progress_reward(None) is None
    assert compose_v2(14, 14, 6, progress=None).progress is None


def test_progress_is_capped_so_task_size_cannot_dominate():
    assert progress_reward(MAX_SMOKE_DELTA) == pytest.approx(1.0)
    assert progress_reward(MAX_SMOKE_DELTA * 100) == pytest.approx(1.0)
    assert progress_reward(MAX_SMOKE_DELTA // 2) == pytest.approx(0.5)


def test_negative_delta_scores_no_progress_rather_than_a_penalty():
    """Regression is already priced by component A; charging twice would double-count it."""
    assert progress_reward(-3) == 0.0


# --- progress: reading the smoke suite ---------------------------------------------------


def test_the_unmeasured_case_is_distinguished_from_the_zero_case():
    assert smoke_delta([]) is None
    assert smoke_delta(["no test counts here"]) is None
    assert smoke_delta(["6/11 passed"]) == 0


def test_progress_is_measured_against_the_first_observation_not_zero():
    """The agent did not author the repo, so the starting count is not progress it earned."""
    summaries = ["6/11 passed", "8/11 passed", "9/11 passed"]
    assert smoke_delta(summaries) == 3


def test_the_best_seen_state_counts_not_the_final_one():
    """An agent that improved, then broke it, keeps the credit for having improved.

    The final damage is already in the submitted patch and is charged by component A. Using
    the last reading instead would make any exploration that is later reverted cost twice,
    which penalises exactly the behaviour the reward is meant to encourage.
    """
    summaries = ["6/11 passed", "10/11 passed", "6/11 passed"]
    assert smoke_delta(summaries) == 4


def test_ending_below_the_baseline_is_not_negative_progress():
    summaries = ["6/11 passed", "4/11 passed"]
    assert smoke_delta(summaries) == 0


def test_the_summary_format_of_run_tests_is_parsed():
    """The exact string `Toolbox.run_tests` produces, including its parenthetical caveat."""
    real = "6/11 passed (smoke tests only; these are not the grading tests)"
    assert smoke_delta([real, "9/11 passed (smoke tests only; these are not the grading tests)"]) == 3


# --- the fused score ----------------------------------------------------------------------


def test_solving_everything_with_no_progress_is_the_correctness_weight():
    assert compose_v2(14, 14, 6, progress=0.0).final == pytest.approx(CORRECTNESS_WEIGHT)


def test_solving_everything_with_perfect_progress_reaches_0_80():
    assert compose_v2(14, 14, 6, progress=1.0).final == pytest.approx(0.80)


def test_doing_nothing_scores_zero_not_the_old_floor():
    """The floor moved to 0.0 in v1 and must stay there in v2."""
    assert compose_v2(6, 14, 6, progress=0.0).final == pytest.approx(0.0)


def test_breaking_what_passed_goes_negative():
    assert compose_v2(4, 14, 6, progress=0.0).final == pytest.approx(-0.15)
    assert compose_v2(0, 14, 6, progress=0.0).final < 0


def test_a_negative_correctness_with_perfect_progress_is_still_negative():
    """Progress cannot make a regressing rollout look like a success."""
    final = compose_v2(4, 14, 6, progress=1.0).final
    assert final < 0


def test_the_fused_score_never_leaves_the_v1_scale():
    for passed in range(0, 15):
        for progress in (None, 0.0, 0.5, 1.0):
            final = compose_v2(passed, 14, 6, progress=progress).final
            assert REWARD_MIN <= final <= REWARD_SOLVED


# --- the trace object ---------------------------------------------------------------------


def test_the_trace_object_matches_the_schema_keys_exactly():
    components = compose_v2(10, 14, 6, progress=0.5).to_components()
    assert set(components) == {"correctness", "progress", "validity", "tool", "final"}


def test_unshipped_components_are_null_not_zero():
    components = compose_v2(10, 14, 6, progress=0.5).to_components()
    assert components["validity"] is None
    assert components["tool"] is None


def test_the_reported_final_is_the_number_the_components_imply():
    components = compose_v2(10, 14, 6, progress=0.5).to_components()
    implied = CORRECTNESS_WEIGHT * components["correctness"] + PROGRESS_WEIGHT * components["progress"]
    assert components["final"] == pytest.approx(implied)


# --- what this buys GRPO ---------------------------------------------------------------


def test_progress_creates_spread_in_a_group_that_was_degenerate():
    """The premise of the whole phase, as an executable claim.

    Four rollouts that all end on the floor under v1 were indistinguishable. With progress
    measured they separate, and the group gains a non-zero standard deviation - which is the
    only thing that produces a gradient.
    """
    v1_group = [0.0, 0.0, 0.0, 0.0]
    assert statistics.pstdev(v1_group) == 0.0

    v2_group = [compose_v2(6, 14, 6, progress=progress_reward(delta)).final for delta in (0, 2, 5, 9)]
    assert statistics.pstdev(v2_group) > 0
    # and the ordering is preserved: more progress, higher reward
    assert v2_group == sorted(v2_group)


def test_progress_alone_cannot_reach_a_solved_rollout_no_progress():
    """Even the best possible progress term is worth less than one solved task."""
    best_progress = compose_v2(6, 14, 6, progress=1.0).final
    solved = compose_v2(14, 14, 6, progress=0.0).final
    assert best_progress < solved
