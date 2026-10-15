"""Convert a NeMo Gym multi-turn rollout into one RL training sample.

One rollout is one GRPO group member. Model-generated turns with captured
generation token IDs are trainable. Tool outputs and synthetic actions without
captured generation tokens remain context only and never receive a gradient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class InvalidRolloutError(ValueError):
    """The rollout contains inconsistent or unsafe training data."""


@dataclass(frozen=True)
class TrainableTurn:
    """One model-generated turn that may receive a gradient."""

    output_index: int
    action_type: str
    action_name: str | None
    prompt_token_ids: tuple[int, ...]
    generation_token_ids: tuple[int, ...]
    generation_log_probs: tuple[float, ...]


@dataclass(frozen=True)
class EpisodeTrainingSample:
    """One scored multi-turn episode and its trainable model turns."""

    task_id: str
    reward: float
    submitted: bool
    turns: tuple[TrainableTurn, ...]
    context_item_count: int


def _require_int_tokens(value: Any, *, field: str, output_index: int) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise InvalidRolloutError(
            f"output[{output_index}].{field} must be a non-empty list"
        )

    if any(isinstance(token, bool) or not isinstance(token, int) for token in value):
        raise InvalidRolloutError(
            f"output[{output_index}].{field} must contain only integer token IDs"
        )

    return tuple(value)


def _require_log_probs(value: Any, *, output_index: int) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise InvalidRolloutError(
            f"output[{output_index}].generation_log_probs must be a non-empty list"
        )

    converted: list[float] = []

    for log_prob in value:
        if isinstance(log_prob, bool) or not isinstance(log_prob, (int, float)):
            raise InvalidRolloutError(
                f"output[{output_index}].generation_log_probs "
                "must contain only numbers"
            )

        converted_log_prob = float(log_prob)

        if not math.isfinite(converted_log_prob):
            raise InvalidRolloutError(
                f"output[{output_index}].generation_log_probs "
                "contains a non-finite value"
            )

        converted.append(converted_log_prob)

    return tuple(converted)


def _require_reward(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidRolloutError("rollout reward must be a number")

    reward = float(value)

    if not math.isfinite(reward):
        raise InvalidRolloutError("rollout reward must be finite")

    return reward


def adapt_rollout(row: dict[str, Any]) -> EpisodeTrainingSample | None:
    """Convert one materialized rollout into one episode training sample.

    Returns ``None`` when the rollout has no trainable model turns. This means
    discard the rollout. It must not be replaced with a fake token or scored as
    zero.

    Raises ``InvalidRolloutError`` when token data exists but is internally
    inconsistent.
    """
    if not isinstance(row, dict):
        raise InvalidRolloutError("rollout row must be a dictionary")

    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise InvalidRolloutError("rollout task_id must be a non-empty string")

    reward = _require_reward(row.get("reward"))

    submitted = row.get("submitted")
    if not isinstance(submitted, bool):
        raise InvalidRolloutError("rollout submitted must be a boolean")

    response = row.get("response")
    if not isinstance(response, dict):
        raise InvalidRolloutError("rollout response must be a dictionary")

    output = response.get("output")
    if not isinstance(output, list):
        raise InvalidRolloutError("rollout response.output must be a list")

    turns: list[TrainableTurn] = []

    for output_index, item in enumerate(output):
        if not isinstance(item, dict):
            raise InvalidRolloutError(
                f"output[{output_index}] must be a dictionary"
            )

        generation_ids = item.get("generation_token_ids")

        # Tool results and synthetic actions without captured model tokens are
        # context only. They must not receive a gradient.
        if not generation_ids:
            continue

        prompt_ids = _require_int_tokens(
            item.get("prompt_token_ids"),
            field="prompt_token_ids",
            output_index=output_index,
        )
        generation_ids_tuple = _require_int_tokens(
            generation_ids,
            field="generation_token_ids",
            output_index=output_index,
        )
        generation_log_probs = _require_log_probs(
            item.get("generation_log_probs"),
            output_index=output_index,
        )

        if len(generation_ids_tuple) != len(generation_log_probs):
            raise InvalidRolloutError(
                f"output[{output_index}] has "
                f"{len(generation_ids_tuple)} generation token IDs but "
                f"{len(generation_log_probs)} generation log probabilities"
            )

        action_type = item.get("type")
        if not isinstance(action_type, str) or not action_type:
            raise InvalidRolloutError(
                f"output[{output_index}].type must be a non-empty string"
            )

        action_name = item.get("name")
        if action_name is not None and not isinstance(action_name, str):
            raise InvalidRolloutError(
                f"output[{output_index}].name must be a string or null"
            )

        turns.append(
            TrainableTurn(
                output_index=output_index,
                action_type=action_type,
                action_name=action_name,
                prompt_token_ids=prompt_ids,
                generation_token_ids=generation_ids_tuple,
                generation_log_probs=generation_log_probs,
            )
        )

    if not turns:
        return None

    return EpisodeTrainingSample(
        task_id=task_id,
        reward=reward,
        submitted=submitted,
        turns=tuple(turns),
        context_item_count=len(output) - len(turns),
    )
