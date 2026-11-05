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


@pytest.fixture(scope="module")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="module")
def proto_text() -> str:
    return PROTOCOL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def trainer_text() -> str:
    return TRAINING_PATH.read_text(encoding="utf-8")


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
    experiment can no longer answer the research question.

    The trained arm's value is now read from its own config rather than assumed absent. It
    used to be asserted that the trainer set no temperature at all, which meant the trained
    arm's sampling was whatever the framework chose - the exact thing Phase 5.4 set out to
    stop relying on. Both arms now name it.

    The path is `policy.generation.temperature`, not `policy.generation.vllm_cfg.temperature`:
    `GenerationConfig` declares it at the top level of the generation mapping and
    `_build_sampling_params` reads it as `self.cfg["temperature"]` (NeMo RL 55296257).
    """
    trainer_temperature = training["policy"]["generation"]["temperature"]
    assert protocol["sampling"]["temperature"] == baseline["sampling"]["temperature"]
    assert protocol["sampling"]["temperature"] == trainer_temperature


def test_sampling_top_p_is_identical_in_both_arms(
    protocol: dict[str, Any], baseline: dict[str, Any], training: dict[str, Any]
) -> None:
    """Nucleus truncation must be the same in the control and the trained arm.

    The protocol recorded `top_p: null` until Phase 5.4 because the baseline set `1.0` while
    the trainer set nothing at all, so "both arms sample identically" rested on an unverified
    framework default. It is now configured in both arms. A drift here reintroduces exactly
    the confound the protocol exists to prevent, so it is asserted directly.

    The path is `policy.generation.top_p`, not `policy.generation.vllm_cfg.top_p`:
    `_build_sampling_params` reads `self.cfg["top_p"]` and `VllmSpecificArgs` does not
    declare `top_p` at all (NeMo RL 55296257).
    """
    trainer_top_p = training["policy"]["generation"]["top_p"]
    assert protocol["sampling"]["top_p"] == baseline["sampling"]["top_p"]
    assert protocol["sampling"]["top_p"] == trainer_top_p
    assert 0.0 < protocol["sampling"]["top_p"] <= 1.0, "top_p outside (0, 1] is not a probability mass"


def test_the_trainer_places_sampling_directly_under_generation(training: dict[str, Any]) -> None:
    """Sampling keys belong on `policy.generation`, not inside `policy.generation.vllm_cfg`.

    This is the structural half of the schema question, grounded in NeMo RL at commit
    55296257c96d49cd95c7d77613cb0f36bd4a4dc7:

      * `GenerationConfig` (nemo_rl/models/generation/interfaces.py) declares
        `temperature: float` and `top_p: float` as top-level members.
      * `VllmConfig(GenerationConfig)` (nemo_rl/models/generation/vllm/config.py) adds
        `vllm_cfg: VllmSpecificArgs`, and `VllmSpecificArgs` declares neither key.
      * `_build_sampling_params` (nemo_rl/models/generation/vllm/vllm_worker.py) reads
        `self.cfg["temperature"]` and `self.cfg["top_p"]`, where `self.cfg` is the
        generation mapping itself - a direct subscript, so a nested key would KeyError
        at the first sampling call rather than be silently ignored.

    This asserts file structure only. It does NOT exercise the NeMo RL parser: that needs
    Python >=3.13.13,<3.14 and an installed NeMo RL, neither of which exists here. Runtime
    acceptance of this config remains unproven and is recorded as such in the trainer file.
    """
    generation = training["policy"]["generation"]
    assert generation["temperature"] == 1.0
    assert generation["top_p"] == 1.0
    assert "temperature" not in generation["vllm_cfg"], (
        "temperature nested under vllm_cfg is not a member of VllmSpecificArgs; it moves up"
    )
    assert "top_p" not in generation["vllm_cfg"], (
        "top_p nested under vllm_cfg is not a member of VllmSpecificArgs; it moves up"
    )


def test_protocol_and_trainer_agree_on_the_documented_sampling_path(proto_text: str, trainer_text: str) -> None:
    """Both files must name the same key path, and the stale nested path must be gone.

    After Phase 5.4 the protocol cited `policy.generation.vllm_cfg.top_p`, which the pinned
    upstream source shows is the wrong container. A document that still points at a path
    the trainer no longer uses is worse than one that says nothing, so the citation is
    asserted to match the resolved path.
    """
    assert "policy.generation.top_p" in proto_text or "policy.generation`" in proto_text, (
        "the protocol must cite the resolved sampling path"
    )
    assert "policy.generation.vllm_cfg.top_p" not in proto_text, "the protocol still cites the superseded nested path"
    assert "grounded" in trainer_text.lower() or "GROUNDED" in trainer_text, (
        "the trainer must cite its grounding source for the sampling keys"
    )
    assert "55296257" in trainer_text, "the trainer must name the pinned commit it was checked against"


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
        ("grpo.kl_coefficient", "there is no reference policy, so no KL term"),
    ],
)
def test_known_undecided_values_stay_explicitly_null(protocol: dict[str, Any], dotted: str, reason: str) -> None:
    """A null is an honest gap; a plausible number would be a false claim.

    Only `grpo.kl_coefficient` is still null after Phase 5.4. It is not *undecided* in the
    sense of "someone must choose a number" - the quantity does not exist in the current
    implementation - but it stays null rather than 0.0 so that the manifest keeps reporting
    it and a reader must look at `kl_coefficient_status` to learn why.

    `model.revision` and `sampling.top_p` were removed from this list because they are now
    resolved; they are covered by the tests below instead.
    """
    assert _get(protocol, dotted) is None, f"{dotted} is undecided: {reason}"


def test_model_revision_is_pinned_to_a_commit_not_a_branch(protocol: dict[str, Any]) -> None:
    """A repo name resolves to whatever is current; a sha does not move.

    Phase 5.4 pinned this to the revision of the snapshot verified against the local Hugging
    Face cache (see the provenance comment in the protocol file). The shape is asserted
    rather than the literal value: a full 40-hex sha, and specifically not `main`, because
    `main` is the failure mode this field exists to prevent.
    """
    revision = protocol["model"]["revision"]
    assert isinstance(revision, str) and revision
    assert revision != "main"
    assert len(revision) == 40, "a full commit sha, not an abbreviation that can collide"
    int(revision, 16)  # raises if any character is not a hex digit


def test_top_p_is_actually_set_rather_than_inherited(protocol: dict[str, Any]) -> None:
    """The protocol must name a value; a null would mean a framework default decides.

    This is the specific gap Phase 5.4 closed. It is deliberately separate from the
    cross-arm parity test: parity can hold while both arms are wrong, but a non-null here
    means somebody chose it and wrote down that they had.
    """
    assert protocol["sampling"]["top_p"] is not None


def test_kl_null_is_declared_not_applicable_rather_than_simply_missing(protocol: dict[str, Any]) -> None:
    """A null with no status is indistinguishable from an oversight.

    The KL coefficient is the one remaining null. `status` is what separates "closed
    question - this quantity does not exist" from the open gaps that had to be resolved
    before a pilot could run, and it is what lets a reader tell that a run is not blocked.
    """
    assert protocol["grpo"]["kl_coefficient"] is None
    assert protocol["grpo"]["kl_coefficient_status"] == "NOT_APPLICABLE_CURRENT_IMPLEMENTATION"
    assert protocol["grpo"]["kl_coefficient_reason"]


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
