"""Phase 5.4 - the rollout-only pilot cannot quietly become something else.

`configs/training/pilot_rollouts_1p5b.yaml` describes a run that collects traces and
performs no update. That is a safety property enforced by a file, and a file can be edited.
These tests are the part that cannot be edited in the same commit without notice.

Three things could go wrong, and all three are silent:

  * the optimiser gets enabled, so a "pilot" spends GPU hours training and its traces stop
    describing the base model;
  * a task or split is named that was never authorised, or the sealed held-out split
    becomes reachable, breaking `docs/heldout_seal.md`;
  * a sampling value drifts away from the formal protocol, so the pilot's timings and score
    distribution describe a run the experiment will not reproduce.

Everything here reads YAML. No GPU, no torch, no model, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_PATH = REPO_ROOT / "configs" / "training" / "pilot_rollouts_1p5b.yaml"
PROTOCOL_PATH = REPO_ROOT / "configs" / "training" / "protocol_1p5b.yaml"


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@pytest.fixture(scope="module")
def pilot() -> dict[str, Any]:
    return _load(PILOT_PATH)


@pytest.fixture(scope="module")
def protocol() -> dict[str, Any]:
    return _load(PROTOCOL_PATH)


# --- it is not a training run ------------------------------------------------------------


def test_the_pilot_performs_no_parameter_update(pilot: dict[str, Any]) -> None:
    """Zero iterations, not "a small number". A step is a step.

    This is the defining property. If it changes, every claim the pilot supports changes
    with it, and the traces stop being evidence about the base model.
    """
    assert pilot["training"]["training_iterations"] == 0
    assert pilot["training"]["optimizer_enabled"] is False
    assert pilot["purpose"]["performs_parameter_update"] is False


def test_the_pilot_does_not_expect_or_declare_an_output_checkpoint(pilot: dict[str, Any]) -> None:
    """A run with no optimiser produces no weights; expecting some would fail a good pilot."""
    assert pilot["training"]["expect_output_checkpoint"] is False
    assert "checkpoint_every_n_iterations" not in pilot["training"]


def test_the_lora_block_is_recorded_as_inert(pilot: dict[str, Any]) -> None:
    """The adapter shape is carried for comparability; the block must say it is not applied."""
    assert pilot["lora"]["applied"] is False


def test_no_advantage_estimator_is_applied(pilot: dict[str, Any]) -> None:
    """GRPO's advantage needs a group of rewards; a pilot computes none."""
    assert pilot["grpo"]["advantage_estimator"] == "not_applied"
    assert pilot["grpo"]["clip_epsilon"] is None


# --- it cannot reach the sealed split -----------------------------------------------------


def test_the_pilot_never_enables_the_heldout_split(pilot: dict[str, Any]) -> None:
    """`docs/heldout_seal.md`: the held-out split is evaluated once, against a frozen checkpoint.

    A pilot is a training-side activity and is on that side of the seal.
    """
    assert pilot["data"]["heldout_enabled"] is False
    assert pilot["data"]["split"] == "train"
    assert pilot["data"]["validation_from"] == "train"


def test_the_pilot_does_not_name_a_task_or_split_it_was_not_given(pilot: dict[str, Any]) -> None:
    """`tasks: null` is the honest record: no task has been authorised for the pilot yet.

    Asserting the null rather than skipping the check is the point. A later edit that fills
    this with a guessed task ID would satisfy "a task is named" while naming one nobody
    approved, and this test makes that edit visible instead.
    """
    assert pilot["rollouts"]["tasks"] is None
    assert pilot["data"]["produces_result"] is False


# --- it samples the way the experiment will -----------------------------------------------


def test_the_pilot_samples_identically_to_the_formal_protocol(pilot: dict[str, Any], protocol: dict[str, Any]) -> None:
    """A pilot whose sampler differs measures a run the experiment will not reproduce."""
    assert pilot["sampling"]["temperature"] == protocol["sampling"]["temperature"]
    assert pilot["sampling"]["top_p"] == protocol["sampling"]["top_p"]
    assert pilot["sampling"]["seed"] == protocol["sampling"]["seed"]
    assert pilot["sampling"]["top_p"] is not None, "an inherited default is not a setting"


def test_the_pilot_loads_the_same_weights_as_the_formal_protocol(
    pilot: dict[str, Any], protocol: dict[str, Any]
) -> None:
    """If these diverge, the pilot validated a different model than the experiment uses.

    The revision is compared as well as the name: a repo name can resolve to different
    weights over time, which is the whole reason the revision is pinned.
    """
    assert pilot["model"]["name"] == protocol["model"]["name"]
    assert pilot["model"]["revision"] == protocol["model"]["revision"]
    assert pilot["model"]["revision"] is not None
    assert pilot["model"]["load_in_4bit"] == protocol["model"]["load_in_4bit"]
    assert pilot["model"]["max_seq_length"] == protocol["model"]["max_seq_length"]


def test_the_pilot_uses_the_reward_the_runtime_actually_scores_with(
    pilot: dict[str, Any], protocol: dict[str, Any]
) -> None:
    """A pilot is what reward_v2 is later computed offline over; the boundary must be explicit."""
    assert pilot["reward"]["live"] == protocol["reward"]["live"] == "reward_v1"
    assert pilot["reward"]["offline_only"] == "reward_v2"
    assert pilot["reward"]["reward_v2_computed_inline"] is False
    assert pilot["reward"]["range"] == protocol["reward"]["range"]


# --- it is a finite, single-GPU, self-cleaning run -----------------------------------------


def test_the_pilot_is_serial_and_shuts_down_afterwards(pilot: dict[str, Any]) -> None:
    """A leaked vLLM process holds VRAM and turns the next run's launch into a false OOM."""
    assert pilot["hardware"]["concurrency"] == 1
    assert pilot["hardware"]["shutdown_required"] is True
    assert pilot["hardware"]["require_gpu"] is True


def test_the_pilot_bounds_every_episode_in_wall_clock_time(pilot: dict[str, Any]) -> None:
    """Without a ceiling, one hung tool call consumes the whole pilot slot.

    Bounding it is not hardening for later: detecting exactly this is part of what a pilot
    is for, so the bound belongs in the pilot.
    """
    assert pilot["episode"]["timeout_s_per_rollout"] > 0
    assert pilot["episode"]["tool_call_timeout_s"] > 0
    assert pilot["episode"]["max_turns"] == "from_task_yaml"
    assert pilot["episode"]["max_tool_calls"] == "from_task_yaml"


def test_the_pilot_names_one_device_explicitly(pilot: dict[str, Any]) -> None:
    """Phase 5.3: GPU 0 only, set rather than relied upon - the variable remaps, not filters."""
    assert pilot["hardware"]["cuda_visible_devices"] == "0"
    assert "," not in pilot["hardware"]["cuda_visible_devices"]


def test_the_kl_null_is_not_a_pilot_blocker(pilot: dict[str, Any]) -> None:
    """The one remaining null is a closed question, not an undecided value.

    Recorded so that "the pilot has a null in it" cannot be read as "the pilot is blocked".
    """
    assert pilot["grpo"]["kl_coefficient"] is None
    assert pilot["grpo"]["kl_coefficient_status"] == "NOT_APPLICABLE_CURRENT_IMPLEMENTATION"


# --- it is declared as a protocol, not a trainer config ------------------------------------


def test_the_pilot_declares_itself_a_protocol_and_never_executed(pilot: dict[str, Any]) -> None:
    """Same marker the safety scan uses, and the same status the arm configs carry."""
    assert pilot["meta"]["kind"] == "arm_independent_protocol"
    assert pilot["meta"]["status"] == "NEVER_EXECUTED"
    assert pilot["meta"]["name"] == "pilot_rollouts_1p5b"
    assert pilot["meta"]["formal_protocol"] == "configs/training/protocol_1p5b.yaml"


def test_the_pilot_does_not_duplicate_the_formal_protocol_file(pilot: dict[str, Any]) -> None:
    """Two files with the same name would make the hash in a manifest ambiguous."""
    assert pilot["meta"]["name"] != _load(PROTOCOL_PATH)["meta"]["name"]
