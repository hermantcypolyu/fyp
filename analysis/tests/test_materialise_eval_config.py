"""Tests for `scripts/materialise_eval_config.py`.

The materialiser exists because the collector resolves no `extends:`. That makes its merge
rule load-bearing: if it silently drops a key, the run executes a config nobody reviewed, and
if it silently *adds* one, the run departs from the base it claims to inherit. Both failures
are invisible in a diff of the output file, so they are pinned here instead.
"""

from __future__ import annotations

import pytest
import yaml

from scripts.materialise_eval_config import MaterialiseError, _deep_merge, materialise, validate

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def test_a_scalar_in_the_overlay_replaces_the_base_value() -> None:
    merged = _deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}}


def test_a_list_is_replaced_wholesale_rather_than_concatenated() -> None:
    """Merging lists would let a one-task overlay silently re-run every task in the base."""
    merged = _deep_merge({"run": {"task_subset": ["a", "b", "c"]}}, {"run": {"task_subset": ["c"]}})
    assert merged["run"]["task_subset"] == ["c"]


def test_extend_is_consumed_and_never_copied_into_the_output() -> None:
    """The materialised file must not still claim to extend something; it is self-contained."""
    merged = _deep_merge({"x": 1}, {"extends": "base.yaml", "x": 2})
    assert "extends" not in merged
    assert merged["x"] == 2


def test_the_base_is_not_mutated() -> None:
    """`_load` returns a fresh dict per call, but a shared reference here would poison it."""
    base = {"a": {"b": 1}}
    _deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}


def test_the_phase6g_overlay_materialises_to_the_phase6e_settings() -> None:
    """Phase 6G must differ from 6E only in run identity and output paths."""
    merged = materialise(REPO_ROOT / "configs" / "eval" / "phase6g_diagnostic_smoke.yaml")
    settings = validate(merged)

    assert settings["model_name"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert settings["model_revision"] == "2e1fd397ee46e1388853d2af2c993145b0f1098a"
    assert settings["temperature"] == 1.0
    assert settings["top_p"] == 1.0
    assert settings["seed"] == 42
    assert settings["split"] == "train"
    assert settings["tasks"] == ["bugfix_offbyone_pagination"]
    assert settings["rollouts_per_task"] == 4
    assert settings["diagnostic_only"] is True
    assert settings["reward_live"] == "reward_v1"


def test_materialising_refuses_a_config_that_would_run_the_heldout_split() -> None:
    base = yaml.safe_load((REPO_ROOT / "configs" / "eval" / "baseline_phase6.yaml").read_text(encoding="utf-8"))
    base["tasks"]["split"] = "heldout"
    with pytest.raises(MaterialiseError, match="sealed"):
        validate(base)


def test_materialising_refuses_a_config_with_a_lora_adapter() -> None:
    """A diagnostic smoke runs the base model; an adapter would change what is measured."""
    base = yaml.safe_load((REPO_ROOT / "configs" / "eval" / "baseline_phase6.yaml").read_text(encoding="utf-8"))
    base["model"]["adapter"] = "outputs/some_adapter"
    with pytest.raises(MaterialiseError, match="base model"):
        validate(base)


def test_materialising_refuses_to_merge_onto_a_base_the_overlay_does_not_name() -> None:
    with pytest.raises(MaterialiseError, match="refusing to"):
        materialise(
            REPO_ROOT / "configs" / "eval" / "phase6g_diagnostic_smoke.yaml",
            base_path=REPO_ROOT / "configs" / "training" / "protocol_1p5b.yaml",
        )
