"""Phase 6L: the 3B capacity diagnostic pins model size and nothing else.

Phase 6I, 6J and 6K each changed one harness lever, each was implemented correctly, and
none of them moved the reward. Phase 6K concluded that the harness is no longer limiting
and that the remaining candidates are model capacity and task design. Phase 6L tests
capacity: it swaps the 1.5B base for the 3B base and holds every other decision fixed.

These tests exist because the arm is only interpretable if model size is the *only*
difference. A capacity comparison that also silently relaxes a turn budget, a sampling
setting or a reward version measures two things at once and can attribute either effect
to the wrong cause. Every assertion here is therefore negative: it pins a value that
must equal the Phase 6K value rather than a value that is merely plausible.

The design rule: this file asserts the overlay against the FROZEN baseline and against the
Phase 6K overlay. It never asserts that the baseline itself changed - the baseline is
immutable and a test that required it to change would have to edit it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = REPO_ROOT / "configs" / "eval" / "baseline_phase6.yaml"
PHASE6K = REPO_ROOT / "configs" / "eval" / "phase6k_workflow_procedure.yaml"
PHASE6L = REPO_ROOT / "configs" / "eval" / "phase6l_model_capacity_3b.yaml"

MODEL_1P5B = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_3B = "Qwen/Qwen2.5-Coder-3B-Instruct"
REVISION_1P5B = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
REVISION_3B = "488639f1ff808d1d3d0ba301aef8c11461451ec5"

# The four seeds Phase 6K derived by sha256_u32_v1 from base seed 42 and task
# `bugfix_offbyone_pagination`. They are restated so the comparison is over the same four
# episodes, not merely over four episodes chosen by the same rule.
PHASE6K_SEEDS = (184288048, 3168762815, 475852200, 1693580453)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _overlay() -> dict[str, Any]:
    return _load(PHASE6L)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """The same rule `scripts/materialise_eval_config.py` applies.

    Re-stated rather than imported: the script is a `__main__` entry point, and importing it
    would couple a config test to a script's importable surface for no benefit.
    """
    out = dict(base)
    for key, value in over.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _arm() -> dict[str, Any]:
    """The Phase 6L arm exactly as the collector will read it: overlay onto baseline."""
    return _deep_merge(_load(BASELINE), _overlay())


# ------------------------------------------------------------------ the one intentful change
def test_the_arm_targets_the_3b_model() -> None:
    """The whole point of the arm: a larger model of the same family and tuning."""
    assert _overlay()["model"]["name"] == MODEL_3B
    assert _arm()["model"]["name"] == MODEL_3B


def test_the_3b_revision_is_a_full_immutable_sha() -> None:
    """A repo name alone resolves to whatever is current, which is invisible irreproducibility."""
    revision = _overlay()["model"]["revision"]
    assert isinstance(revision, str)
    assert len(revision) == 40, "revision must be a full git sha"
    assert revision == REVISION_3B
    assert all(c in "0123456789abcdef" for c in revision)
    assert revision != REVISION_1P5B, "the arm must not silently pin the 1.5B weights"


def test_the_frozen_baseline_still_pins_the_1p5b_model() -> None:
    """Phase 6L is an overlay. Editing the baseline would invalidate the Phase 6 numbers."""
    assert _load(BASELINE)["model"]["name"] == MODEL_1P5B
    assert _load(BASELINE)["model"]["revision"] == REVISION_1P5B


def test_the_arm_loads_no_adapter() -> None:
    """A LoRA would make a capacity comparison into a capacity-plus-adapter comparison."""
    assert _arm()["model"]["adapter"] is None


def test_the_arm_does_not_quantise_the_weights() -> None:
    """4-bit changes the weights, so it would measure quantisation rather than parameter count."""
    arm = _arm()
    assert arm["model"]["load_in_4bit"] is False
    assert arm["model"]["dtype"] == "bfloat16"


def test_the_context_window_is_unchanged() -> None:
    """Both arms must see the same window, or the token budget is a second variable."""
    assert _arm()["model"]["max_seq_length"] == _load(BASELINE)["model"]["max_seq_length"] == 8192


# ------------------------------------------------------------------- sampling: held fixed
@pytest.mark.parametrize(
    "key,expected",
    [
        ("temperature", 1.0),
        ("top_p", 1.0),
        ("seed", 42),
        ("seed_policy", "per_rollout"),
        ("seed_derivation", "sha256_u32_v1"),
    ],
)
def test_sampling_settings_match_the_phase_6k_arm(key: str, expected: object) -> None:
    """Resampling settings would change the episodes, not the model's ability to handle them."""
    assert _arm()["sampling"][key] == expected


def test_the_four_derived_seeds_are_the_phase_6k_seeds() -> None:
    """The comparison is over the same four episodes, recomputed from the same rule."""
    from analysis.sampling_seeds import derive_rollout_seed

    seeds = tuple(
        derive_rollout_seed(base_seed=42, task_id="bugfix_offbyone_pagination", rollout_index=i) for i in range(4)
    )
    assert seeds == PHASE6K_SEEDS


# ---------------------------------------------------------------------- run: held fixed
def test_the_arm_runs_exactly_four_rollouts_of_one_task() -> None:
    arm = _arm()["run"]
    assert arm["rollouts_per_task"] == 4
    assert arm["task_subset"] == ["bugfix_offbyone_pagination"]
    assert arm["min_valid_rollouts"] == 4


def test_the_arm_is_a_diagnostic_arm() -> None:
    """`diagnostic_only` is a veto in `_assess_validity`: this arm can never be training evidence."""
    assert _arm()["run"]["diagnostic_only"] is True


def test_the_arm_uses_the_same_task_as_phase_6k() -> None:
    assert _load(PHASE6K)["run"]["task_subset"] == _arm()["run"]["task_subset"]


# --------------------------------------------------------------- step budgets: held fixed
def test_turn_and_tool_budgets_stay_inherited_from_the_task_yaml() -> None:
    """A larger model must not be given more turns; that would confound capacity with budget."""
    arm = _arm()["episode"]
    assert arm["max_turns"] == "from_task_yaml"
    assert arm["max_tool_calls"] == "from_task_yaml"
    # And the overlay must not mention them at all, so they cannot drift by omission either.
    assert "episode" not in _overlay()


def test_the_arm_does_not_change_the_turn_cap_in_the_overlay() -> None:
    """Belt and braces: no key anywhere in the overlay names a budget."""
    text = PHASE6L.read_text(encoding="utf-8")
    for banned in ("max_turns:", "max_tool_calls:", "max_tokens_per_turn:"):
        assert banned not in text, f"{banned} must not appear in the 3B overlay"


# ------------------------------------------------------------------- reward: held fixed
def test_reward_versions_are_inherited_and_not_restated() -> None:
    """v1 live and v2 analysed. The overlay must not carry its own copy to drift."""
    arm = _arm()["reward"]
    assert arm["live"] == "reward_v1"
    assert arm["analysed"] == "reward_v2"
    assert "reward" not in _overlay()


def test_the_arm_declares_no_optimizer() -> None:
    """The eval config schema has no optimizer key, and the collector never trains."""
    assert "optimizer" not in _overlay()
    assert "optimizer" not in _arm()


def test_heldout_stays_disabled() -> None:
    """Held-out evaluation is a one-shot sealed resource; a diagnostic arm must not touch it."""
    arm = _arm()
    assert arm["tasks"]["heldout_enabled"] is False
    assert arm["tasks"]["split"] == "train"
    # The overlay must not restate it: an inherited false cannot drift, a copied one can.
    assert "tasks" not in _overlay()


# -------------------------------------------------------------------------- the GPU record
def test_the_gpu_override_is_recorded_rather_than_implicit() -> None:
    """Rule 1 says GPU 0. This run was authorised onto physical GPU 1 and the file must say so.

    The override is legitimate but it must be legible: a reader comparing 6K and 6L needs to
    see that the serving device differs and that it was recorded, not inferred.
    """
    overlay = _overlay()
    assert overlay["server"]["cuda_visible_devices"] == "1"
    assert '"' in PHASE6L.read_text(encoding="utf-8").split("cuda_visible_devices:", 1)[1][:3], (
        "the device must be a quoted string so it cannot be parsed as an integer index"
    )
    # The reason must be in the file, not only in a commit message.
    header = PHASE6L.read_text(encoding="utf-8").lower()
    assert "authoris" in header or "authoriz" in header


def test_the_served_model_name_matches_the_arm_model() -> None:
    """vLLM serves one name; a mismatch makes the collector talk to a model it did not pin."""
    arm = _arm()
    assert arm["server"]["served_model_name"] == arm["model"]["name"] == MODEL_3B


def test_the_arm_uses_the_same_port_as_the_control() -> None:
    """Port 8000 belongs to another user on this host; 8010 is the agreed port."""
    assert _arm()["server"]["base_url"] == _load(BASELINE)["server"]["base_url"]


# --------------------------------------------------------------------------- what changed
def test_model_identity_and_run_shape_are_the_only_differences() -> None:
    """Enumerate the full delta against the baseline and require it to be the permitted set.

    This is the guard that makes the arm a capacity comparison. If a future edit adds a
    sixth difference, this test fails and the new difference has to be argued for.
    """
    base = _load(BASELINE)
    arm = _arm()

    def flat(node: Any, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        if isinstance(node, dict):
            for key, value in node.items():
                out.update(flat(value, f"{prefix}{key}."))
        else:
            out[f"{prefix[:-1]}"] = node
        return out

    base_flat, arm_flat = flat(base), flat(arm)
    changed = {k for k in arm_flat if k in base_flat and base_flat[k] != arm_flat[k]}
    added = {k for k in arm_flat if k not in base_flat}

    permitted_changed = {
        "meta.name",
        "meta.phase",
        "meta.role",
        "meta.status",
        "meta.written",
        "model.name",
        "model.revision",
        "run.rollouts_per_task",
        "run.task_subset",
        "run.min_valid_rollouts",
        "server.cuda_visible_devices",
        "server.served_model_name",
    }
    permitted_added = {"run.diagnostic_only"}

    assert changed <= permitted_changed, f"unexpected changes: {sorted(changed - permitted_changed)}"
    assert added <= permitted_added, f"unexpected additions: {sorted(added - permitted_added)}"
    # And the change set must not be empty: an overlay that changes nothing is a dead file.
    assert {"model.name", "model.revision"} <= changed
