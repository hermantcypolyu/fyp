"""Reward composition for the coding environment. See `rewards/compose.py`."""

from rewards.compose import (
    GATE_REWARD,
    REWARD_DID_NOTHING,
    REWARD_MIN,
    REWARD_SOLVED,
    compose_reward,
    normalised_reward,
)

__all__ = [
    "GATE_REWARD",
    "REWARD_DID_NOTHING",
    "REWARD_MIN",
    "REWARD_SOLVED",
    "compose_reward",
    "normalised_reward",
]

