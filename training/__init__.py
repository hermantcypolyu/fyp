"""Training-data adapters for multi-turn coding-agent rollouts."""

from training.rollout_adapter import (
    EpisodeTrainingSample,
    InvalidRolloutError,
    TrainableTurn,
    adapt_rollout,
)

__all__ = [
    "EpisodeTrainingSample",
    "InvalidRolloutError",
    "TrainableTurn",
    "adapt_rollout",
]
