"""Phase 5.1 - the arms share one protocol, and the build notices when they stop.

`configs/training/protocol_1p5b.yaml` is the single place the experiment's settings are
written down. That is only worth something if drifting away from it fails a test rather
than being discovered in the discussion section. Three things can go wrong:

  * an arm config and the protocol disagree (the baseline samples at a different
    temperature than the trained arm, so "RL changed the score" is not attributable);
  * the protocol loses a value Phase 5.1 requires, silently, because a key was renamed;
  * a value is recorded as decided when it is not - a plausible `top_p: 1.0` invented to
    fill a table is worse than a visible `null`.

Each test below is a restatement of one of those, so a failure names the parity rule it
broke. Where a value is genuinely undecided the test asserts it is *declared undecided*
and listed in the manifest, rather than asserting a number.

Everything here reads YAML. No GPU, no torch, no model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "configs" / "training" / "protocol_1p5b.yaml"
BASELINE_PATH = REPO_ROOT / "configs" / "eval" / "baseline.yaml"
TRAINING_PATH = REPO_ROOT / "configs" / "training" / "grpo_nemo_rl_a5000.yaml"

#: Every value Phase 5.1 asks the protocol to record, as a dotted path.
PHASE_5_1_FIELDS = [
    "model.name",  # base model name
    "model.revision",  # base model revision
    "lora.r",  # LoRA rank
    "lora.lora_alpha",  # LoRA alpha
    "lora.target_modules",  # target modules
    "training.learning_rate",  # learning rate
    "rollouts.group_size",  # group size
    "rollouts.n_tasks",  # number of tasks
    "rollouts.rollouts_per_task",  # rollouts per task
    "sampling.temperature",  # temperature
    "sampling.top_p",  # top-p
    "episode.max_turns",  # max turns
    "episode.max_tool_calls",  # max tool calls
    "model.max_seq_length",  # max sequence length
    "training.max_grad_norm",  # gradient clipping
    "grpo.kl_coefficient",  # KL coefficient
    "grpo.clip_epsilon",  # clip epsilon
    "sampling.seed",  # random seed
    "training.training_iterations",  # training iterations
    "training.checkpoint_every_n_iterations",  # checkpoint schedule
]


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@pytest.fixture(scope="module")
def protocol() -> dict[str, Any]:
    return _load(PROTOCOL_PATH)


@pytest.fixture(scope="module")
def baseline() -> dict[str, Any]:
    return _load(BASELINE_PATH)


@pytest.fixture(scope="module")
def training() -> dict[str, Any]:
    return _load(TRAINING_PATH)


def _get(doc: dict[str, Any], dotted: str) -> Any:
    node: Any = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


# --- 5.1 completeness -----------------------------------------------------------------


@pytest.mark.parametrize("dotted", PHASE_5_1_FIELDS)
def test_protocol_declares_every_phase_5_1_setting(protocol: dict[str, Any], dotted: str) -> None:
    """The key must exist.

    A key present with a null value is a *declared gap*; a missing key is an oversight that
    reads identically to "not applicable" in a later review. This test distinguishes them.
    """
    _get(protocol, dotted)


def test_protocol_names_the_model_the_arms_actually_load(
    protocol: dict[str, Any], baseline: dict[str, Any], training: dict[str, Any]
) -> None:
    """A protocol naming a model neither arm loads would be documentation, not a control."""
    name = protocol["model"]["name"]
    assert baseline["model"]["name"] == name
    assert training["policy"]["model_name"] == name


# --- cross-arm parity ------------------------------------------------------------------


def test_sampling_temperature_is_identical_in_both_arms(
    protocol: dict[str, Any], baseline: dict[str, Any], training: dict[str, Any]
) -> None:
    """The control must be sampled the way the policy is sampled during rollouts.

    If these diverge, a difference between the arms is partly a sampler difference, and the
    experiment can no longer answer the research question. Only the baseline is asserted:
    `grpo_nemo_rl_a5000.yaml` does not restate a temperature, so the protocol file is the
    trained arm's only record of it - which is precisely why it is written there.
    """
    assert protocol["sampling"]["temperature"] == baseline["sampling"]["temperature"]
    assert "temperature" not in training.get("grpo", {}), (
        "the active trainer config now sets its own temperature; assert it equals the "
        "protocol rather than leaving two sources of truth"
    )


def test_max_seq_length_is_identical_in_both_arms(protocol: dict[str, Any], baseline: dict[str, Any]) -> None:
    """A different context window means the two arms see different observations.

    Only the baseline is queried: the active NeMo RL config does not repeat the length, so
    for the trained arm the protocol file is the sole authority, and asserting it against
    itself would prove nothing.
    """
    assert protocol["model"]["max_seq_length"] == baseline["model"]["max_seq_length"]


def test_quantisation_is_identical_in_both_arms(protocol: dict[str, Any], baseline: dict[str, Any]) -> None:
    """4-bit vs 16-bit is a bigger difference than RL/no-RL, so it must not vary."""
    assert protocol["model"]["load_in_4bit"] == baseline["model"]["load_in_4bit"]


def test_episode_budget_comes_from_the_task_in_both_arms(protocol: dict[str, Any], baseline: dict[str, Any]) -> None:
    """A task's own budget is authoritative, so neither arm can hard-code a different one."""
    assert protocol["episode"]["max_turns"] == "from_task_yaml"
    assert protocol["episode"]["max_tool_calls"] == "from_task_yaml"
    assert baseline["episode"]["max_turns"] == "from_task_yaml"
    assert baseline["episode"]["max_tool_calls"] == "from_task_yaml"


def test_group_size_matches_the_trainer_config(protocol: dict[str, Any], training: dict[str, Any]) -> None:
    """The protocol's group size must be the one the trainer would actually use.

    GRPO's advantage is a within-group statistic; a protocol group size the trainer does not
    honour would make every variance figure in the analysis describe a different run.
    """
    trainer_group = training["grpo"]["num_generations_per_prompt"]
    assert protocol["rollouts"]["group_size"] == trainer_group
    assert protocol["rollouts"]["rollouts_per_task"] == trainer_group


def test_baseline_samples_the_same_number_as_a_group(protocol: dict[str, Any], baseline: dict[str, Any]) -> None:
    """The control arm's k must equal the trained arm's group size for pass@k to be comparable."""
    assert baseline["sampling"]["n_samples_per_task"] == protocol["rollouts"]["group_size"]


def test_lora_targets_cover_both_projections_of_every_layer(protocol: dict[str, Any]) -> None:
    """Truncating the target list silently trains less of the model than reported."""
    assert set(protocol["lora"]["target_modules"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    assert protocol["lora"]["lora_alpha"] == protocol["lora"]["r"] * 2


def test_reward_versions_are_named_and_the_live_one_is_v1(protocol: dict[str, Any]) -> None:
    """The reward is part of the protocol; a reward change between arms is a confound.

    `reward_v2` is computed offline by `scripts/reward_distribution.py` and is not imported
    by any live path, so the runtime scores with `reward_v1`. If that changes, this test is
    the place the change is registered.
    """
    assert protocol["reward"]["live"] == "reward_v1"
    assert protocol["reward"]["analysed"] == "reward_v2"
    assert protocol["reward"]["range"] == [-1.0, 1.0]


def test_training_never_enables_the_heldout_split(protocol: dict[str, Any], baseline: dict[str, Any]) -> None:
    """`docs/heldout_seal.md`: the held-out split is evaluated once, at the end, and never trained on."""
    assert protocol["data"]["split"] == "train"
    assert protocol["data"]["validation_from"] == "train"
    assert protocol["data"]["heldout_enabled"] is False
    assert baseline["splits"]["heldout"]["enabled"] is False


def test_only_one_gpu_device_is_ever_named(protocol: dict[str, Any]) -> None:
    """Phase 5.3: GPU 0 only, and the variable is set explicitly rather than relied upon."""
    assert protocol["hardware"]["cuda_visible_devices"] == "0"


# --- declared gaps are gaps, not invented values ----------------------------------------


@pytest.mark.parametrize(
    ("dotted", "reason"),
    [
        ("model.revision", "weights are not pinned to a commit"),
        ("sampling.top_p", "no arm config selects one"),
        ("grpo.kl_coefficient", "there is no reference policy, so no KL term"),
    ],
)
def test_known_undecided_values_stay_explicitly_null(protocol: dict[str, Any], dotted: str, reason: str) -> None:
    """A null is an honest gap; a plausible number would be a false claim.

    Filling one of these is a deliberate act that must also update the manifest's
    `unresolved` list, so this test failing is a prompt to do it properly.
    """
    assert _get(protocol, dotted) is None, f"{dotted} is undecided: {reason}"


def test_kl_absence_is_consistent_with_the_phase_4_audit(protocol: dict[str, Any]) -> None:
    """Phase 4 found no reference model and no KL term. The protocol must not imply one.

    Recording `kl_coefficient: 0.0` would claim a term that exists and is switched off -
    a different and stronger claim than "absent".
    """
    assert protocol["grpo"]["kl_coefficient"] is None
    assert protocol["grpo"]["advantage_estimator"] == "group_relative"


def test_protocol_declares_itself_never_executed(protocol: dict[str, Any]) -> None:
    """Mirrors the arm configs. A protocol that ran is a different status, and would say so."""
    assert protocol["meta"]["status"] == "NEVER_EXECUTED"
    assert protocol["meta"]["model_tier"] == "1.5B"
