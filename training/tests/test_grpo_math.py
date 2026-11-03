"""Unit tests for the torch-free GRPO arithmetic in `training.grpo_math`.

These exist because the arithmetic used to live inside
`scripts/run_optimizer_smoke.py`, which imports torch, peft and transformers and
immediately demands a CUDA device. None of it could be executed in CI, so the
group math was only ever verified by a real 1.5B-model run on a GPU box.

The first test below is the important one: it replays the rewards from an
actually-executed smoke run and asserts the results to the last decimal, so this
suite is anchored to measured output rather than to invented examples.
"""

from __future__ import annotations

import math
import statistics

import pytest

from training.grpo_math import (
    ADVANTAGE_EPSILON,
    MAX_LOG_RATIO,
    DegenerateGroupError,
    clipped_surrogate,
    group_advantages,
    policy_loss,
    policy_ratio,
)

# Rewards recorded in results/raw/rl_smoke_1step/training_metrics.json, a real
# optimizer smoke run on an RTX A6000. One rollout tripped a gate (-1.0).
RECORDED_REWARDS = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
RECORDED_MEAN = -0.125
RECORDED_STD = 0.33071891388307384
RECORDED_ADVANTAGE_POSITIVE = 0.37796333015554007
RECORDED_ADVANTAGE_NEGATIVE = -2.6457433110887805


# --- the anchor: reproducibility of a real run ------------------------------------------


def test_reproduces_the_recorded_smoke_run_exactly():
    """The extracted math must be the math that actually executed.

    Every literal here was read out of a metrics file produced by
    `scripts/run_optimizer_smoke.py` on real hardware, not computed by this test
    from the implementation. If the refactor changed a single bit, this fails.
    """
    advantages, mean, std = group_advantages(RECORDED_REWARDS)

    assert mean == RECORDED_MEAN
    assert std == RECORDED_STD
    assert advantages[0] == RECORDED_ADVANTAGE_POSITIVE
    assert advantages[4] == RECORDED_ADVANTAGE_NEGATIVE


# --- 4.4 Equal rewards -> advantage near zero (actually: rejected) -----------------------


def test_equal_rewards_are_rejected_rather_than_scored_as_zero_advantage():
    """A zero-variance group has no advantage - not an advantage of zero.

    Returning zeros would give a loss with no gradient, and the run would look
    like it trained. `DegenerateGroupError` is the whole point of the check.
    """
    with pytest.raises(DegenerateGroupError, match="no learning signal"):
        group_advantages([0.0, 0.0, 0.0, 0.0])

    with pytest.raises(DegenerateGroupError):
        group_advantages([-1.0] * 8)


def test_a_group_smaller_than_two_cannot_have_a_mean():
    with pytest.raises(DegenerateGroupError, match="at least two"):
        group_advantages([0.5])


# --- 4.4 High-reward / low-reward rollouts ----------------------------------------------


def test_high_reward_rollout_gets_positive_advantage():
    advantages, mean, _ = group_advantages([0.0, 0.5, 1.0])
    assert advantages[2] > 0
    assert advantages[2] == max(advantages)
    assert mean == pytest.approx(0.5)


def test_low_reward_rollout_gets_negative_advantage():
    advantages, _, _ = group_advantages([0.0, 0.5, 1.0])
    assert advantages[0] < 0
    assert advantages[0] == min(advantages)


def test_the_lowest_reward_always_gets_the_lowest_advantage():
    """Advantage must be monotone in reward, or the gradient points the wrong way."""
    rewards = [0.0, 0.1, -0.7, 0.9, 0.3]
    advantages, _, _ = group_advantages(rewards)
    by_reward = sorted(range(len(rewards)), key=lambda index: rewards[index])
    assert [advantages[index] for index in by_reward] == sorted(advantages)


def test_advantages_are_centred_and_scaled():
    """Standardisation is what makes the scale of the reward irrelevant."""
    advantages, _, _ = group_advantages([0.0, 100.0, 250.0, 900.0])

    assert statistics.fmean(advantages) == pytest.approx(0.0, abs=1e-12)
    # Slightly below 1 because of ADVANTAGE_EPSILON in the denominator - the
    # recorded run shows exactly 0.9999969762933588, not 1.0.
    assert statistics.pstdev(advantages) < 1.0
    assert statistics.pstdev(advantages) > 0.99


def test_advantage_is_invariant_to_an_affine_rescale_of_the_rewards():
    """Scale and offset of the reward drop out. Only the ordering survives.

    This holds only to within `ADVANTAGE_EPSILON`: the denominator is
    `std + eps`, so a group whose rescaled standard deviation differs in the
    last bits gets a very slightly different normaliser. The disagreement is
    bounded by the epsilon itself, not by float noise.
    """
    original, _, _ = group_advantages([0.0, 0.4, 1.0])
    scaled, _, _ = group_advantages([-50.0 + 30.0 * reward for reward in [0.0, 0.4, 1.0]])

    for left, right in zip(original, scaled, strict=True):
        assert left == pytest.approx(right, abs=1e-5)


def test_non_finite_and_non_numeric_rewards_are_rejected():
    for bad in ([0.0, math.nan], [0.0, math.inf], [0.0, -math.inf], [0.0, True], [0.0, "1.0"]):
        with pytest.raises(DegenerateGroupError):
            group_advantages(bad)


# --- 4.4 Clipping actually bounds the ratio ---------------------------------------------


def test_ratio_is_exp_of_the_log_difference():
    ratios = policy_ratio([-1.0], [-1.5])
    assert ratios[0] == pytest.approx(math.exp(0.5))


def test_a_giant_log_difference_is_clamped_instead_of_overflowing():
    """An unclamped exp of 1000 is what turns a loss into `inf`."""
    assert policy_ratio([1000.0], [-1000.0])[0] == pytest.approx(math.exp(MAX_LOG_RATIO))
    assert math.isfinite(policy_ratio([1000.0], [-1000.0])[0])


def test_clipping_limits_the_reward_for_improving_too_far():
    """Above `1+e` the objective stops rising, so the update stops chasing it."""
    ratios = [1.5, 2.0, 10.0]
    values = [clipped_surrogate(ratio, 1.0, 0.2) for ratio in ratios]
    assert all(value == pytest.approx(1.2) for value in values)


def test_clipping_does_not_apply_to_the_disadvantageous_direction():
    """The PPO `min` is deliberately one-sided.

    When the advantage is negative, an *increased* ratio is penalised at full
    weight - clipping that would let a bad update escape the trust region.
    """
    assert clipped_surrogate(3.0, -1.0, 0.2) == pytest.approx(-3.0)


def test_inside_the_trust_region_the_objective_is_not_clipped():
    for ratio in (0.85, 1.0, 1.15):
        assert clipped_surrogate(ratio, 1.0, 0.2) == pytest.approx(ratio)


def test_clip_epsilon_must_be_non_negative():
    with pytest.raises(ValueError, match="non-negative"):
        clipped_surrogate(1.0, 1.0, -0.2)


# --- 4.4 KL is absent, and the loss must not pretend otherwise ---------------------------


def test_there_is_no_kl_term_to_test():
    """Component 4.4 asks that "rising KL is reflected in the loss". It cannot be.

    `scripts/run_optimizer_smoke.py` loads no reference model and computes no KL
    divergence, and `policy_loss` takes no KL input. This test pins that absence
    so the checklist item cannot be quietly ticked later: if a `kl` parameter is
    added, this test fails and whoever added it is prompted to add the real test.
    """
    import inspect

    from training import grpo_math

    parameters = inspect.signature(grpo_math.policy_loss).parameters
    assert "kl" not in parameters
    assert "kl_coefficient" not in parameters
    assert not hasattr(grpo_math, "kl_divergence")
    assert not hasattr(grpo_math, "reference_log_probs")


# --- policy loss shape -------------------------------------------------------------------


def test_loss_is_negated_because_the_surrogate_is_maximised():
    """A positive advantage must produce a negative loss, since we minimise."""
    loss = policy_loss([1.0], [1.0], clip_epsilon=0.2, token_count=1)
    assert loss == pytest.approx(-1.0)


def test_loss_is_normalised_by_the_group_token_count_not_the_turn_count():
    """The divisor is the group token count, so adding tokens scales the total.

    Dividing by a turn count instead would weight an episode by how many turns it
    happened to take, which is the efficiency measure that was explicitly
    rejected for training.
    """
    one_token = policy_loss([1.0], [1.0], clip_epsilon=0.2, token_count=2)
    two_tokens = policy_loss([1.0, 1.0], [1.0, 1.0], clip_epsilon=0.2, token_count=2)

    # Same divisor, twice the numerator.
    assert two_tokens == pytest.approx(2 * one_token)
    assert one_token == pytest.approx(-0.5)


def test_per_call_losses_sum_to_the_group_loss():
    """Splitting a group into per-turn calls must not change the total."""
    ratios, advantages = [1.0, 0.9, 1.1], [1.0, -1.0, 0.5]
    whole = policy_loss(ratios, advantages, clip_epsilon=0.2, token_count=3)

    parts = math.fsum(
        policy_loss([ratio], [advantage], clip_epsilon=0.2, token_count=3)
        for ratio, advantage in zip(ratios, advantages, strict=True)
    )

    assert whole == pytest.approx(parts)


def test_zero_advantage_contributes_no_gradient_signal():
    """The whole point of removing the baseline: a mean reward teaches nothing."""
    assert policy_loss([1.0, 2.0], [0.0, 0.0], clip_epsilon=0.2, token_count=2) == pytest.approx(0.0)


# --- 4.4 Loss, gradient, KL all finite ---------------------------------------------------


def test_loss_is_finite_for_every_extreme_ratio_and_advantage():
    for ratio in (0.0, 1e-9, 0.5, 1.0, 1.0 + 1e-9, 1e9, float(MAX_LOG_RATIO)):
        for advantage in (-1e6, -1.0, 0.0, 1.0, 1e6):
            loss = policy_loss([ratio], [advantage], clip_epsilon=0.2, token_count=1)
            assert math.isfinite(loss), (ratio, advantage)


def test_policy_ratio_output_is_always_finite():
    for current, old in ((0.0, 0.0), (-1e6, 1e6), (1e6, -1e6), (-0.5, -0.4)):
        ratios = policy_ratio([current], [old])
        assert all(math.isfinite(ratio) and ratio > 0 for ratio in ratios), (current, old)


# --- 4.4 Empty / invalid rollout safely rejected ----------------------------------------


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="equal length"):
        policy_ratio([1.0, 2.0], [1.0])

    with pytest.raises(ValueError, match="align"):
        policy_loss([1.0, 2.0], [1.0], clip_epsilon=0.2, token_count=2)


def test_non_finite_log_probabilities_are_rejected():
    with pytest.raises(ValueError, match="not a finite number"):
        policy_ratio([math.nan], [0.0])
    with pytest.raises(ValueError, match="not a finite number"):
        policy_ratio([0.0], [math.inf])


def test_zero_token_count_is_rejected_rather_than_dividing_by_zero():
    with pytest.raises(ValueError, match="token_count must be positive"):
        policy_loss([1.0], [1.0], clip_epsilon=0.2, token_count=0)


def test_empty_inputs_are_handled_without_a_divide_by_zero():
    """An empty episode contributes exactly nothing, not a NaN."""
    assert policy_loss([], [], clip_epsilon=0.2, token_count=5) == pytest.approx(0.0)


def test_the_epsilon_keeps_the_scale_just_under_one():
    """Pins why the recorded run reports advantage_std 0.9999969762933588, not 1.0."""
    advantages, _, _ = group_advantages([0.0, 1.0])
    assert statistics.pstdev(advantages) == pytest.approx(
        1.0 / (1.0 + ADVANTAGE_EPSILON / statistics.pstdev([-0.5, 0.5]))
    )
