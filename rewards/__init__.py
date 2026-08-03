"""Reward composition for the coding environment. See `rewards/compose.py`."""

from rewards.compose import (
    GATE_REWARD,
    REWARD_DID_NOTHING,
    REWARD_MIN,
    REWARD_SOLVED,
    compose_reward,
    normalised_reward,
)
from rewards.transport import (
    DiscardRateTooHigh,
    DiscardTracker,
    VerifyTransportError,
    reward_from_verify_response,
    score_or_discard,
)

__all__ = [
    "GATE_REWARD",
    "REWARD_DID_NOTHING",
    "REWARD_MIN",
    "REWARD_SOLVED",
    "DiscardRateTooHigh",
    "DiscardTracker",
    "VerifyTransportError",
    "compose_reward",
    "normalised_reward",
    "reward_from_verify_response",
    "score_or_discard",
]
