"""The torch-free arithmetic of a GRPO group, extracted so it can be tested.

`scripts/run_optimizer_smoke.py` needs a GPU, a 1.5B model and ~20 GB of VRAM to
import, which means none of its arithmetic could be executed in CI. The parts
that are pure Python - advantage normalisation and the clipped surrogate
objective - are therefore defined here and imported back by the script, so the
numbers that were tested are the numbers that run.

Nothing in this module imports torch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

#: Guard added to the reward standard deviation. A group whose rewards differ by
#: less than this is treated as degenerate rather than divided by ~zero.
ADVANTAGE_EPSILON = 1.0e-6

#: `exp(log_ratio)` is clamped to this range before use. A log-ratio of 20 is a
#: probability ratio of ~4.85e8 - far beyond anything PPO clipping can act on,
#: and well past the point where `float32` `exp` overflows to `inf`.
MAX_LOG_RATIO = 20.0


class DegenerateGroupError(ValueError):
    """The group carries no learning signal, so it cannot produce an advantage.

    GRPO computes `(reward - group_mean) / group_std`. When every member has the
    same reward the numerator and denominator are both zero - not a small
    advantage, but no gradient at all. Silently returning zeros would produce a
    loss with no gradient and the run would look like it trained. This is raised
    instead.
    """


def group_advantages(rewards: Sequence[float]) -> tuple[list[float], float, float]:
    """Normalise rewards into group-relative advantages.

    Returns `(advantages, mean, standard_deviation)`. The advantages are
    standardised within the group, so they always have mean ~0 and standard
    deviation ~1: only the *relative* ordering of the group matters, which is
    the whole point of removing the value baseline.

    Raises `DegenerateGroupError` when the group is smaller than two, contains a
    non-finite reward, or has zero spread.
    """
    if len(rewards) < 2:
        raise DegenerateGroupError(f"A GRPO group needs at least two members to have a mean; got {len(rewards)}.")

    for index, reward in enumerate(rewards):
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise DegenerateGroupError(f"Reward at index {index} is not a number: {reward!r}")
        if not math.isfinite(reward):
            raise DegenerateGroupError(f"Reward at index {index} is not finite: {reward!r}")

    reward_mean = math.fsum(rewards) / len(rewards)
    # Population standard deviation: the group *is* the population here, we are
    # not sampling an estimate of a larger spread.
    reward_std = math.sqrt(math.fsum((reward - reward_mean) ** 2 for reward in rewards) / len(rewards))

    if reward_std <= 0:
        raise DegenerateGroupError(
            "Every member of this group has the same reward, so there is no learning signal. "
            "This is the zero-variance failure, not a group that happens to be easy."
        )

    advantages = [(reward - reward_mean) / (reward_std + ADVANTAGE_EPSILON) for reward in rewards]

    if not all(math.isfinite(advantage) for advantage in advantages):
        raise DegenerateGroupError("Advantage normalisation produced a non-finite value.")

    return advantages, reward_mean, reward_std


def policy_ratio(current_log_probs: Sequence[float], old_log_probs: Sequence[float]) -> list[float]:
    """`exp(current - old)`, the importance-sampling ratio `pi_theta / pi_old`.

    The log-ratio is clamped to `[-MAX_LOG_RATIO, MAX_LOG_RATIO]` before
    exponentiation so a pathological pair of log-probabilities cannot overflow to
    `inf` and poison the gradient. The clamp sits so far outside the PPO trust
    region (`±0.2` by default) that it never binds on a real update.
    """
    if len(current_log_probs) != len(old_log_probs):
        raise ValueError(
            f"Current and old log-probabilities must have equal length; "
            f"got {len(current_log_probs)} and {len(old_log_probs)}."
        )

    ratios: list[float] = []

    for index, (current, old) in enumerate(zip(current_log_probs, old_log_probs, strict=True)):
        for name, value in (("current", current), ("old", old)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} log-probability at index {index} is not a finite number: {value!r}")

        log_ratio = max(-MAX_LOG_RATIO, min(MAX_LOG_RATIO, current - old))
        ratios.append(math.exp(log_ratio))

    return ratios


def clipped_surrogate(ratio: float, advantage: float, clip_epsilon: float) -> float:
    """The PPO/GRPO clipped objective for one token: `min(r*A, clip(r)*A)`.

    The `min` against the *unclipped* term is what makes the objective a
    pessimistic lower bound. Without it the ratio could be pushed arbitrarily
    far in the direction that helps; with it, once the ratio leaves
    `[1-e, 1+e]` the gradient through that term becomes zero and the update stops
    chasing an advantage the trust region has already spent.
    """
    if clip_epsilon < 0:
        raise ValueError(f"clip_epsilon must be non-negative; got {clip_epsilon}.")

    clipped_ratio = max(1.0 - clip_epsilon, min(1.0 + clip_epsilon, ratio))

    return min(ratio * advantage, clipped_ratio * advantage)


def policy_loss(
    ratios: Sequence[float], advantages: Sequence[float], *, clip_epsilon: float, token_count: int
) -> float:
    """Token-weighted masked policy loss for one episode.

    Negated because the optimiser minimises and the surrogate is a quantity to
    *maximise*. Divided by the **total** token count of the group, not the turns
    in this episode, so that concatenating every per-episode call reproduces the
    single group loss: no episode is weighted by how many turns it happened to
    take.
    """
    if len(ratios) != len(advantages):
        raise ValueError(f"ratios and advantages must align; got {len(ratios)} and {len(advantages)}.")
    if token_count <= 0:
        raise ValueError(f"token_count must be positive; got {token_count}.")

    total = math.fsum(
        clipped_surrogate(ratio, advantage, clip_epsilon) for ratio, advantage in zip(ratios, advantages, strict=True)
    )

    return -total / token_count
