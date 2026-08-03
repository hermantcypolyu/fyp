"""Turning a `/verify` HTTP response into a reward - or into nothing at all.

A rollout that could not be *scored* is not a rollout that scored *badly*. Under ADR 0006
`0.0` is a real, meaningful score - it is exactly what a do-nothing agent earns - so a
failed or non-200 `/verify` that returns `0.0` injects a fake do-nothing sample into the
group. It is indistinguishable from a genuine one, it moves the group mean and standard
deviation, and it raises no error. The run looks healthy the whole time.

The upstream pattern makes this easy to walk into. Both reference implementations default
the reward to zero on failure:

    # unslothai/notebooks, NeMo-Gym-Multi-Environment.ipynb
    reward = resp.json().get("reward", 0.0) if resp.status_code == 200 else 0.0

    # NVIDIA-NeMo/Gym, responses_api_agents/verifiers_agent/app.py
    reward = rollout_output.get("reward", 0.0) or 0.0

So the rule here is deliberately blunt: **a transport failure never produces a number.**
`reward_from_verify_response` raises, and `score_or_discard` converts that into `None`.

`None` is not an invention either - it is TRL's own contract for this case. From TRL's
OpenEnv guide, on `rollout_reward_fn`:

    Return a `float`, or return `None` to mark the rollout unscorable so it is dropped
    from the group baseline instead of being counted as a zero.

A dropped rollout is cheap; a fabricated one is not. But dropping silently is its own
hazard, so `DiscardTracker` records the rate every step and fails loudly past a threshold -
a discard rate climbing toward 100% means the environment is down, and training on whatever
survives is worse than stopping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DiscardRateTooHigh",
    "DiscardTracker",
    "VerifyTransportError",
    "reward_from_verify_response",
    "score_or_discard",
]


class VerifyTransportError(RuntimeError):
    """`/verify` did not return a usable reward. Never carries a fallback score."""


class DiscardRateTooHigh(RuntimeError):
    """Too many rollouts were unscorable. Training on the remainder is not sound."""


def reward_from_verify_response(
    *,
    status_code: int,
    payload: Any,
    task_id: str = "",
) -> float:
    """Extract the reward from a `/verify` response, or raise.

    Raises `VerifyTransportError` on anything that is not an unambiguous numeric reward.
    It has no default and no fallback: returning a number here on a broken response is the
    entire failure mode this module exists to prevent.
    """
    where = f" (task {task_id})" if task_id else ""

    if status_code != 200:
        raise VerifyTransportError(f"/verify returned HTTP {status_code}{where}; the rollout is unscorable")
    if not isinstance(payload, dict):
        raise VerifyTransportError(
            f"/verify returned {type(payload).__name__}, expected a JSON object{where}; the rollout is unscorable"
        )
    if "reward" not in payload:
        raise VerifyTransportError(
            f"/verify response has no 'reward' key{where} (keys: {sorted(payload)}); the rollout is unscorable"
        )

    reward = payload["reward"]
    # bool is an int subclass, and `True` would silently become 1.0.
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise VerifyTransportError(
            f"/verify returned reward={reward!r} of type {type(reward).__name__}{where}; the rollout is unscorable"
        )
    if not math.isfinite(reward):
        raise VerifyTransportError(f"/verify returned a non-finite reward ({reward}){where}; the rollout is unscorable")

    return float(reward)


def score_or_discard(
    *,
    status_code: int,
    payload: Any,
    task_id: str = "",
    tracker: DiscardTracker | None = None,
) -> float | None:
    """`reward_from_verify_response`, but returning `None` instead of raising.

    `None` means *drop this rollout from the group* - TRL's contract for an unscorable
    rollout. It does not mean zero, and it must never be coerced to zero downstream.
    """
    try:
        reward = reward_from_verify_response(status_code=status_code, payload=payload, task_id=task_id)
    except VerifyTransportError as exc:
        if tracker is not None:
            tracker.record_discard(str(exc))
        return None
    if tracker is not None:
        tracker.record_scored()
    return reward


@dataclass
class DiscardTracker:
    """Counts unscorable rollouts and refuses to let the run limp along quietly.

    Dropping a rollout is the right response to a transport failure, but a *rising* discard
    rate is a dead environment, and the surviving rollouts are a biased sample of it. Log
    `summary()` every step; call `check()` to stop the run.
    """

    #: Fraction of rollouts that may be discarded before the run is considered unsound.
    max_discard_rate: float = 0.02
    #: Below this many observations the rate is too noisy to act on.
    min_samples: int = 20

    scored: int = 0
    discarded: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.scored + self.discarded

    @property
    def rate(self) -> float:
        return self.discarded / self.total if self.total else 0.0

    def record_scored(self) -> None:
        self.scored += 1

    def record_discard(self, reason: str) -> None:
        self.discarded += 1
        self.reasons.append(reason)

    def check(self) -> None:
        """Raise once the discard rate is both meaningful and too high."""
        if self.total < self.min_samples or self.rate <= self.max_discard_rate:
            return
        recent = "\n  ".join(self.reasons[-5:])
        raise DiscardRateTooHigh(
            f"{self.discarded}/{self.total} rollouts ({self.rate:.1%}) were unscorable, "
            f"above the {self.max_discard_rate:.1%} threshold. The environment is probably "
            f"down; the rollouts that did score are a biased sample of it, so training on "
            f"them is worse than stopping.\n  Most recent reasons:\n  {recent}"
        )

    def summary(self) -> dict[str, float | int]:
        """One row per step for the training log."""
        return {
            "rollouts_scored": self.scored,
            "rollouts_discarded": self.discarded,
            "discard_rate": round(self.rate, 4),
        }

    def reset(self) -> None:
        self.scored = 0
        self.discarded = 0
        self.reasons.clear()
