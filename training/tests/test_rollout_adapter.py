from __future__ import annotations

import copy

import pytest

from training.rollout_adapter import InvalidRolloutError, adapt_rollout


def make_row() -> dict:
    return {
        "task_id": "task_example",
        "reward": 0.0,
        "submitted": True,
        "response": {
            "output": [
                {
                    "type": "function_call",
                    "name": "read_file",
                    "prompt_token_ids": [10, 11],
                    "generation_token_ids": [20, 21],
                    "generation_log_probs": [-0.1, -0.2],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"ok": true}',
                },
                {
                    "type": "function_call",
                    "name": "run_tests",
                },
            ]
        },
    }


def test_adapts_one_rollout_as_one_episode():
    sample = adapt_rollout(make_row())

    assert sample is not None
    assert sample.task_id == "task_example"
    assert sample.reward == 0.0
    assert sample.submitted is True
    assert len(sample.turns) == 1
    assert sample.context_item_count == 2

    turn = sample.turns[0]
    assert turn.output_index == 0
    assert turn.action_type == "function_call"
    assert turn.action_name == "read_file"
    assert turn.prompt_token_ids == (10, 11)
    assert turn.generation_token_ids == (20, 21)
    assert turn.generation_log_probs == (-0.1, -0.2)


def test_preserves_negative_reward_without_clamping():
    row = make_row()
    row["reward"] = -1.0

    sample = adapt_rollout(row)

    assert sample is not None
    assert sample.reward == -1.0


def test_preserves_real_zero_reward():
    sample = adapt_rollout(make_row())

    assert sample is not None
    assert sample.reward == 0.0


def test_tool_outputs_do_not_become_trainable_turns():
    sample = adapt_rollout(make_row())

    assert sample is not None
    assert len(sample.turns) == 1
    assert all(turn.action_type != "function_call_output" for turn in sample.turns)


def test_synthetic_action_without_tokens_is_context_only():
    sample = adapt_rollout(make_row())

    assert sample is not None
    assert tuple(turn.action_name for turn in sample.turns) == ("read_file",)
    assert sample.context_item_count == 2


def test_rollout_without_trainable_tokens_is_discarded():
    row = make_row()

    for item in row["response"]["output"]:
        item.pop("prompt_token_ids", None)
        item.pop("generation_token_ids", None)
        item.pop("generation_log_probs", None)

    assert adapt_rollout(row) is None


def test_rejects_mismatched_token_and_logprob_lengths():
    row = make_row()
    row["response"]["output"][0]["generation_log_probs"] = [-0.1]

    with pytest.raises(InvalidRolloutError, match="token IDs"):
        adapt_rollout(row)


def test_rejects_non_integer_generation_token_ids():
    row = make_row()
    row["response"]["output"][0]["generation_token_ids"] = [20, "21"]

    with pytest.raises(InvalidRolloutError, match="integer token IDs"):
        adapt_rollout(row)


def test_rejects_non_integer_prompt_token_ids():
    row = make_row()
    row["response"]["output"][0]["prompt_token_ids"] = [10, True]

    with pytest.raises(InvalidRolloutError, match="integer token IDs"):
        adapt_rollout(row)


def test_rejects_non_finite_logprob():
    row = make_row()
    row["response"]["output"][0]["generation_log_probs"] = [
        -0.1,
        float("nan"),
    ]

    with pytest.raises(InvalidRolloutError, match="non-finite"):
        adapt_rollout(row)


def test_rejects_missing_reward():
    row = make_row()
    row.pop("reward")

    with pytest.raises(InvalidRolloutError, match="reward"):
        adapt_rollout(row)


def test_adapter_does_not_mutate_original_rollout():
    row = make_row()
    original = copy.deepcopy(row)

    adapt_rollout(row)

    assert row == original
