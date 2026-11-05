"""Sequence length: one protocol budget, three runtime fields that are NOT interchangeable.

The project states a single arm-independent budget, `model.max_seq_length: 8192`, in
`configs/training/protocol_1p5b.yaml`. NeMo RL has no key spelled `model.max_seq_length`,
so that budget has to be mapped onto the field that actually consumes it - and NeMo RL
has three different length fields, each bounding a different thing:

  ``policy.max_total_sequence_length``
      TOTAL tokens per trajectory (prompt + completion).  ``PolicyConfig`` in
      ``nemo_rl/models/policy/__init__.py`` at commit ``55296257``; consumed in
      ``nemo_rl/algorithms/grpo.py`` as ``max_seq_len=`` at every rollout call site.
  ``policy.generation.max_new_tokens``
      COMPLETION-ONLY tokens per request.  ``GenerationConfig`` in
      ``nemo_rl/models/generation/interfaces.py``; consumed in ``vllm_worker.py``
      ``_build_sampling_params`` as ``SamplingParams(max_tokens=...)``.
  ``policy.generation.vllm_cfg.max_model_len``
      ENGINE context capacity.  ``VllmSpecificArgs`` in
      ``nemo_rl/models/generation/vllm/config.py``; passed straight into
      ``vllm.LLM(max_model_len=...)``.

Writing 8192 into the wrong one of those is a real defect that no syntax check catches,
because ``policy`` is a raw ``TypedDict`` inside a Pydantic model declared
``extra="allow"``. A misplaced key loads cleanly and then either does nothing or raises a
``KeyError`` at the first sampling call.

**These are structural and source-grounded tests, not runtime parser tests.** They read
YAML and cross-check it against citations recorded in the config and in this docstring.
NeMo RL is not installed here (it needs Python ``>=3.13.13,<3.14``; this workspace is
3.12, and ``import nemo_rl`` raises ``ModuleNotFoundError``). Nothing in this file proves
that the pinned loader accepts the keys it checks - that remains an open, blocking item.

Everything here reads YAML and text. No GPU, no torch, no model, no NeMo RL import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPO_ROOT / "configs" / "training" / "protocol_1p5b.yaml"
BASELINE_PATH = REPO_ROOT / "configs" / "eval" / "baseline.yaml"
PILOT_PATH = REPO_ROOT / "configs" / "training" / "pilot_rollouts_1p5b.yaml"
TRAINING_PATH = REPO_ROOT / "configs" / "training" / "grpo_nemo_rl_a5000.yaml"

#: The one number every arm is supposed to agree on.
EXPECTED_BUDGET = 8192


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
def pilot() -> dict[str, Any]:
    return _load(PILOT_PATH)


@pytest.fixture(scope="module")
def training() -> dict[str, Any]:
    return _load(TRAINING_PATH)


@pytest.fixture(scope="module")
def trainer_text() -> str:
    return TRAINING_PATH.read_text(encoding="utf-8")


def _policy(training: dict[str, Any]) -> dict[str, Any]:
    policy = training.get("policy")
    assert isinstance(policy, dict), "grpo_nemo_rl_a5000.yaml lost its `policy` mapping"
    return policy


def _generation(training: dict[str, Any]) -> dict[str, Any]:
    generation = _policy(training).get("generation")
    assert isinstance(generation, dict), "`policy.generation` is missing"
    return generation


# --- 1. the protocol states the budget -------------------------------------------------


def test_protocol_states_the_sequence_budget(protocol: dict[str, Any]) -> None:
    """The arm-independent budget exists and is the documented number."""
    assert protocol["model"]["max_seq_length"] == EXPECTED_BUDGET


# --- 2. the trainer's TOTAL budget equals the protocol's -------------------------------


def test_trainer_maps_the_budget_to_max_total_sequence_length(
    protocol: dict[str, Any], training: dict[str, Any]
) -> None:
    """The protocol budget lands on the field that bounds prompt + completion.

    `max_total_sequence_length` is the NeMo RL field consumed as `max_seq_len=` by every
    rollout call site in `grpo.py`. It is where a total trajectory budget belongs.
    """
    assert _policy(training)["max_total_sequence_length"] == protocol["model"]["max_seq_length"]


# --- 3. engine capacity is at least the budget -----------------------------------------


def test_engine_capacity_covers_the_total_budget(training: dict[str, Any]) -> None:
    """vLLM cannot emit a trajectory longer than its own context window.

    `vllm_worker.py` asserts `response_length <= ...max_model_len` after generation, so a
    smaller `max_model_len` makes a full-budget trajectory unrepresentable rather than
    merely slow. The relation, not the number, is what matters here.
    """
    max_model_len = _generation(training)["vllm_cfg"]["max_model_len"]
    assert max_model_len >= _policy(training)["max_total_sequence_length"], (
        f"max_model_len={max_model_len} cannot hold a "
        f"max_total_sequence_length={_policy(training)['max_total_sequence_length']} trajectory"
    )


# --- 4. max_new_tokens is not used as the total budget ---------------------------------


def test_max_new_tokens_is_not_mistaken_for_the_total_budget(training: dict[str, Any]) -> None:
    """`max_new_tokens` caps generation only, so it must not be set to the total budget.

    Setting it to 8192 would be harmless-looking and wrong in two ways: it claims the
    protocol defines a completion budget it does not define, and it asks vLLM to generate
    up to the entire context window on every single request. The field may be absent
    entirely (upstream's default applies) or smaller than the total - never equal to it.
    """
    generation = _generation(training)
    if "max_new_tokens" not in generation:
        return  # unset is the intended state: the protocol defines no completion budget
    assert generation["max_new_tokens"] < _policy(training)["max_total_sequence_length"], (
        "max_new_tokens is a completion-only cap; equal to the total budget it would claim a "
        "completion budget the protocol does not define"
    )


# --- 5. max_new_tokens lives at the right depth ----------------------------------------


def test_completion_cap_is_not_nested_under_vllm_cfg(training: dict[str, Any]) -> None:
    """`max_new_tokens` is a sampler setting, so it belongs at `policy.generation`.

    `GenerationConfig` declares it there and `_build_sampling_params` reads it as
    `self.cfg["max_new_tokens"]`, where `self.cfg` is the `generation` mapping. Nesting it
    under `vllm_cfg` would be silently ignored.
    """
    assert "max_new_tokens" not in _generation(training).get("vllm_cfg", {})


# --- 6. the pilot carries the budget where an equivalent field exists ------------------


def test_pilot_carries_the_budget_without_a_trainer_mapping(pilot: dict[str, Any]) -> None:
    """The pilot is not a NeMo RL run, so `model.max_seq_length` *is* its sequence field.

    There is no `policy.max_total_sequence_length` for it to set - the pilot performs no
    parameter update and constructs no `PolicyConfig`. The budget maps to itself, and the
    absence of a trainer mapping is correct rather than missing.
    """
    assert pilot["model"]["max_seq_length"] == EXPECTED_BUDGET
    assert "policy" not in pilot, "the pilot configures no trainer; a `policy` block would make it look like one"


# --- 7. the baseline carries the budget where an equivalent field exists ---------------


def test_baseline_carries_the_budget_and_has_no_trainer_fields(baseline: dict[str, Any]) -> None:
    """The control arm states the same budget in the field its own path reads.

    Like the pilot, the baseline is not driven by NeMo RL, so `model.max_seq_length` is the
    whole story. It must not grow a `policy` block: an arm that quietly acquires trainer
    fields stops being a clean control.
    """
    assert baseline["model"]["max_seq_length"] == EXPECTED_BUDGET
    assert "policy" not in baseline
    assert "max_new_tokens" not in baseline.get("model", {})


# --- 8. the mapping's provenance is recorded, not remembered ---------------------------


@pytest.mark.parametrize(
    "symbol",
    [
        "max_total_sequence_length",
        "max_new_tokens",
        "max_model_len",
        "PolicyConfig",
        "VllmSpecificArgs",
        "GenerationConfig",
    ],
)
def test_trainer_config_cites_the_source_symbols_for_its_length_fields(trainer_text: str, symbol: str) -> None:
    """Each length field's provenance is written next to it, with a file and a symbol.

    Reading a value without its source is how `max_new_tokens` gets confused with a total
    budget. The config names the declaring class for each of the three fields so that a
    future editor can re-check the mapping without reconstructing it.
    """
    assert symbol in trainer_text, (
        f"{TRAINING_PATH.name} no longer mentions {symbol}; the sequence-length mapping "
        "must cite the pinned source it came from"
    )


def test_trainer_records_that_runtime_parity_is_unproven(trainer_text: str) -> None:
    """The config must not read as if a real loader had accepted it.

    These tests are structural. If the config ever stops saying so, the file starts
    implying a verification that has not happened.
    """
    assert "RUNTIME PARITY IS NOT PROVEN" in trainer_text
