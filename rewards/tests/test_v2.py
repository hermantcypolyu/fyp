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

import math
import statistics

import pytest

from rewards.compose import GATE_REWARD, REWARD_DID_NOTHING, REWARD_MIN, REWARD_SOLVED
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


# --- invariant 1: no double counting between components ---------------------------------
#
# Each observation must feed exactly one component. If a raw quantity reached two
# components, the same fact would be paid for twice and the declared weights would stop
# meaning what they say. These tests pin the separation rather than the arithmetic.


def test_changing_only_correctness_moves_component_a_and_leaves_progress_alone():
    """Hold progress fixed, sweep correctness: B must not move at all."""
    progress_values = []
    for passed in (6, 8, 10, 14):
        result = compose_v2(passed, 14, 6, progress=0.5)
        progress_values.append(result.progress)
        assert result.correctness == pytest.approx((passed - 6) / (14 - 6))
    assert progress_values == [0.5, 0.5, 0.5, 0.5]


def test_changing_only_progress_moves_component_b_and_leaves_correctness_alone():
    """Hold correctness fixed, sweep progress: A must not move at all."""
    correctness_values = []
    for delta in (0, 2, 5, 10):
        result = compose_v2(10, 14, 6, progress=progress_reward(delta))
        correctness_values.append(result.correctness)
    assert correctness_values == [0.5, 0.5, 0.5, 0.5]


def test_the_two_components_are_the_only_addends_of_the_non_gated_total():
    """The total is exactly `A*cw + B*pw` - no term is counted a second time.

    Weights are recovered from the function itself rather than from the constants, so
    this test fails if someone adds a hidden additive term with either sweep.
    """
    for passed, progress in ((14, 0.0), (14, 1.0), (10, 0.5), (6, 0.5)):
        result = compose_v2(passed, 14, 6, progress=progress)
        reconstructed = CORRECTNESS_WEIGHT * result.correctness + PROGRESS_WEIGHT * result.progress
        assert result.final == pytest.approx(reconstructed)


def test_one_observation_feeds_exactly_one_component():
    """A `run_tests` summary is a progress reading and must never touch correctness."""
    baseline, better = "6/11 passed (smoke tests only)", "9/11 passed (smoke tests only)"
    delta = smoke_delta([baseline, better])
    assert delta == 3

    # The same hidden-test outcome, once with the observation and once without.
    observed = compose_v2(8, 14, 6, progress=progress_reward(delta))
    unobserved = compose_v2(8, 14, 6, progress=None)
    assert observed.correctness == unobserved.correctness
    # ...and the whole difference between them is the progress term alone.
    assert observed.final - unobserved.final == pytest.approx(PROGRESS_WEIGHT * observed.progress)


def test_a_gate_zeroes_the_final_but_preserves_the_diagnostic_components():
    """Gate discard is not a second deduction on a component - it replaces the total.

    If gating also zeroed the components, the same penalty would be applied twice and a
    post-mortem could not tell "the model was bad" from "a guard fired".
    """
    gated = compose_v2(14, 14, 6, gates=("tests_modified",), progress=1.0)
    ungated = compose_v2(14, 14, 6, progress=1.0)

    assert gated.final == GATE_REWARD
    # the components are untouched by the gate...
    assert (gated.correctness, gated.progress) == (ungated.correctness, ungated.progress)
    # ...and the final is the gate value, not the productive total minus a penalty.
    assert gated.final != ungated.final - 1.0


# --- invariant 2: the reward is always finite -------------------------------------------
#
# A NaN or an inf silently destroys a gradient and can poison a whole training step. The
# public API forbids the degenerate denominators (a non-failable task), so those are
# treated as contract violations - but every input the API *does* admit must be finite.


def test_final_is_finite_for_every_admissible_combination():
    """Exhaustive over gates x progress x correctness on a real task shape."""
    saw_values = 0
    for passed in range(0, 15):
        for progress in (None, 0.0, 0.25, 0.5, 1.0):
            for gates in ((), ("tests_modified",), ("timeout", "no_submit")):
                result = compose_v2(passed, 14, 6, gates=gates, progress=progress)
                assert math.isfinite(result.final), (passed, progress, gates)
                assert math.isfinite(result.correctness), (passed, progress, gates)
                if result.progress is not None:
                    assert math.isfinite(result.progress), (passed, progress, gates)
                saw_values += 1
    assert saw_values == 15 * 5 * 3


def test_very_large_valid_counts_stay_finite():
    """A big task must not overflow into inf at the ends of the scale."""
    huge = 10**6
    solved = compose_v2(huge, huge, 0, progress=1.0)
    nothing = compose_v2(0, huge, 0, progress=0.0)
    for result in (solved, nothing):
        assert math.isfinite(result.final)
        assert math.isfinite(result.correctness)


def test_extreme_progress_inputs_stay_finite_and_bounded():
    """`smoke_delta` is floored at 0 and `progress_reward` is capped, so neither overflows."""
    assert progress_reward(10**9) == pytest.approx(1.0)
    assert progress_reward(-(10**9)) == pytest.approx(0.0)
    assert math.isfinite(smoke_delta([f"{c}/{10**6} passed" for c in (0, 10**6)]))
    assert smoke_delta([f"{c}/{10**6} passed" for c in (0, 10**6)]) == 10**6


def test_an_unmeasured_component_is_none_and_never_a_numeric_zero():
    """`None` means "not measured"; 0.0 means "measured as no change". They differ."""
    unmeasured = compose_v2(14, 14, 6, progress=None)
    measured_zero = compose_v2(14, 14, 6, progress=0.0)

    assert unmeasured.progress is None
    assert not isinstance(unmeasured.progress, float)
    assert measured_zero.progress == pytest.approx(0.0)
    # Both are finite, and the distinction survives into the trace-facing view.
    assert math.isfinite(unmeasured.final) and math.isfinite(measured_zero.final)
    assert unmeasured.to_components()["progress"] is None
    assert measured_zero.to_components()["progress"] == pytest.approx(0.0)


def test_the_degenerate_denominator_is_a_contract_violation_not_a_nan():
    """A task with no movable tests is rejected outright rather than scoring NaN.

    `validate_tasks.py` rejects such a task as NOT_FAILABLE, so raising here is the
    documented behaviour and the only way a non-finite value could appear is prevented.
    """
    for n_tests, floor_tests in ((0, 0), (4, 4), (4, 6)):
        with pytest.raises(ValueError, match="must exceed floor_tests"):
            compose_v2(0, n_tests, floor_tests)


# --- invariant 3: the reward stays within the declared range ----------------------------
#
# Two different ranges have to be kept apart, and conflating them is a documentation bug:
#
#   declared outer range : [REWARD_MIN, REWARD_SOLVED] = [-1.0, 1.0]
#                          the clamp on the fused value; the scale the reward lives on
#   reachable v2 range   : [-1.0, 0.80]
#                          the extremes actually attainable with only A and B shipped
#
# Every admissible rollout is asserted against the outer range; the reachable extremes are
# asserted separately so that shipping C/D later extends the reachable range on purpose
# rather than by accident.


def test_every_final_value_stays_within_the_declared_outer_range():
    for passed in range(0, 15):
        for progress in (None, 0.0, 0.5, 1.0):
            for gates in ((), ("tests_modified",)):
                final = compose_v2(passed, 14, 6, gates=gates, progress=progress).final
                assert REWARD_MIN <= final <= REWARD_SOLVED, (passed, progress, gates, final)


def test_a_gate_reaches_exactly_the_declared_minimum():
    """The gate value is the floor of the scale, not merely near it."""
    for gates in (("tests_modified",), ("timeout",), ("sandbox_escape", "syntax_broken")):
        assert compose_v2(14, 14, 6, gates=gates, progress=1.0).final == REWARD_MIN
    assert GATE_REWARD == REWARD_MIN


def test_the_reachable_maximum_is_the_sum_of_the_shipped_weights():
    """Not 1.0: with A and B only, the best attainable score is 0.60 + 0.20 = 0.80."""
    best = compose_v2(14, 14, 6, progress=1.0)
    assert best.final == pytest.approx(0.80)
    assert best.final < REWARD_SOLVED
    assert best.final == pytest.approx(sum(WEIGHTS.values()))


def test_the_reachable_range_boundaries():
    """Pin both ends of the reachable range and the interior landmarks between them."""
    landmarks = {
        "gated": (compose_v2(14, 14, 6, gates=("tests_modified",), progress=1.0).final, REWARD_MIN),
        "worst correct, no progress": (compose_v2(0, 14, 6, progress=0.0).final, -0.45),
        "did nothing": (compose_v2(6, 14, 6, progress=0.0).final, REWARD_DID_NOTHING),
        "solved, no measurement": (compose_v2(14, 14, 6, progress=None).final, CORRECTNESS_WEIGHT),
        "solved, zero progress": (compose_v2(14, 14, 6, progress=0.0).final, CORRECTNESS_WEIGHT),
        "solved, perfect progress": (compose_v2(14, 14, 6, progress=1.0).final, 0.80),
    }
    for label, (observed, expected) in landmarks.items():
        assert observed == pytest.approx(expected), label
        assert REWARD_MIN <= observed <= REWARD_SOLVED, label


def test_negative_correctness_stays_above_the_declared_minimum():
    """Breaking what already passed is penalised, but cannot break the floor.

    A rollout that destroys pre-existing passes scores below zero, yet the worst it can do
    is the clamp - it never runs off the bottom of the declared scale.
    """
    broken = compose_v2(0, 14, 6, progress=0.0)
    assert broken.correctness == pytest.approx(-0.75)
    assert broken.final == pytest.approx(-0.45)
    assert broken.final > REWARD_MIN

    # The extreme version of the same thing, across every progress value.
    for progress in (None, 0.0, 0.5, 1.0):
        final = compose_v2(0, 14, 6, progress=progress).final
        assert REWARD_MIN < final <= REWARD_SOLVED


def test_the_reachable_range_is_narrower_than_the_declared_range():
    """The two ranges are documented as different; this keeps them from collapsing."""
    reachable_low = min(
        compose_v2(passed, 14, 6, gates=gates, progress=progress).final
        for passed in range(0, 15)
        for progress in (None, 0.0, 0.5, 1.0)
        for gates in ((), ("tests_modified",))
    )
    reachable_high = max(
        compose_v2(passed, 14, 6, progress=progress).final
        for passed in range(0, 15)
        for progress in (None, 0.0, 0.5, 1.0)
    )
    assert reachable_low == pytest.approx(REWARD_MIN)
    assert reachable_high == pytest.approx(0.80)
    assert reachable_high < REWARD_SOLVED
