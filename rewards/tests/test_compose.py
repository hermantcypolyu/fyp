"""Unit tests for the normalised reward and the gate constant (ADR 0006).

The test that matters most is `test_a_gated_rollout_scores_below_doing_nothing`. Normalising
the reward and moving the gate constant are one change: with the formula in place but gates
still at 0.0, writing to `tests/**` and submitting immediately would score *identically*, and
nothing else in the system would notice - no exception, no failing test, and the gate-rate
metric still logging hits while the reward ignored them. That test is the tripwire.
"""

from __future__ import annotations

import pytest

from rewards.compose import (
    GATE_REWARD,
    REWARD_DID_NOTHING,
    REWARD_MIN,
    REWARD_SOLVED,
    compose_reward,
    normalised_reward,
)

# --- the coupling invariant -------------------------------------------------------------


@pytest.mark.parametrize(
    ("n_tests", "floor_tests"),
    [(13, 5), (17, 9), (9, 0), (14, 8), (10, 5)],
)
def test_a_gated_rollout_scores_below_doing_nothing(n_tests, floor_tests):
    """gate_reward < normalised_reward(p = floor_tests), on every task shape."""
    doing_nothing = normalised_reward(floor_tests, n_tests, floor_tests)
    assert doing_nothing == REWARD_DID_NOTHING
    assert GATE_REWARD < doing_nothing


def test_the_gate_constant_is_the_bottom_of_the_scale():
    assert GATE_REWARD == REWARD_MIN
    assert REWARD_MIN < REWARD_DID_NOTHING < REWARD_SOLVED


def test_a_gate_beats_no_amount_of_real_work():
    # even a fully solved run is discarded once a gate fires
    assert compose_reward(14, 14, 6, gates=["tests_modified"]) == GATE_REWARD
    assert compose_reward(14, 14, 6, gates=[]) == REWARD_SOLVED


# --- the formula ------------------------------------------------------------------------


def test_doing_nothing_scores_exactly_zero():
    assert normalised_reward(6, 14, 6) == 0.0


def test_solving_everything_scores_exactly_one():
    assert normalised_reward(14, 14, 6) == 1.0


def test_partial_progress_is_the_fraction_of_movable_tests():
    # 8 movable tests; four of them won
    assert normalised_reward(10, 14, 6) == pytest.approx(0.5)
    assert normalised_reward(8, 14, 6) == pytest.approx(0.25)
    assert normalised_reward(13, 14, 6) == pytest.approx(0.875)


def test_a_zero_floor_reduces_to_the_raw_pass_fraction():
    assert normalised_reward(3, 9, 0) == pytest.approx(1 / 3)


def test_breaking_what_already_passed_scores_negative():
    assert normalised_reward(4, 14, 6) == pytest.approx(-0.25)
    assert normalised_reward(0, 14, 6) < 0


def test_catastrophic_regression_is_clamped_to_the_floor_of_the_scale():
    # 2 movable tests, 10 already passing: raw would be -5.0
    assert normalised_reward(0, 12, 10) == REWARD_MIN


def test_a_task_with_no_movable_tests_is_rejected():
    with pytest.raises(ValueError, match="NOT_FAILABLE"):
        normalised_reward(9, 9, 9)


# --- negative rewards are safe for a group-normalised advantage --------------------------


def test_negative_rewards_leave_the_advantage_well_defined():
    """GRPO's advantage is (r - mean) / std, which is sign-agnostic.

    What matters is that the wider range cannot produce a NaN or a zero-variance group where
    the raw scale would not have. A group mixing a gate with real work has *more* spread than
    the same group scored raw, which is the point.
    """
    import statistics

    group = [
        compose_reward(6, 14, 6),                          # did nothing      -> 0.0
        compose_reward(10, 14, 6),                         # half the work    -> 0.5
        compose_reward(14, 14, 6),                         # solved           -> 1.0
        compose_reward(14, 14, 6, gates=["tests_modified"]),  # hacked         -> -1.0
    ]
    assert group == [0.0, 0.5, 1.0, -1.0]

    mean = statistics.fmean(group)
    std = statistics.pstdev(group)
    assert std > 0
    advantages = [(r - mean) / std for r in group]
    assert all(a == a for a in advantages)  # no NaN
    # the hacked rollout is the most negative advantage in the group
    assert advantages[-1] == min(advantages)


def test_an_all_gated_group_is_still_degenerate_but_not_nan():
    """Every-member-identical groups have no gradient on any scale; that is not new."""
    import statistics

    group = [compose_reward(3, 14, 6, gates=["no_submit"]) for _ in range(4)]
    assert group == [GATE_REWARD] * 4
    assert statistics.pstdev(group) == 0.0
