"""Keep hazard 2 shut: the trainer must not flatten the bottom of the reward scale.

ADR 0006 widened `reward_v1` to [-1, 1] so that "made it worse" is distinguishable from
"did nothing", and so that every hard gate - `tests_modified` above all - lands strictly
below an immediate submit. That coupling only holds if a `-1.0` survives the trainer
untouched. Several knobs can quietly destroy it, and none of them raise: the run completes,
the loss looks sane, and the agent learns that deleting the hidden tests costs nothing.

Trainer selection was resolved to **NeMo RL** on 2026-08-03. That decision is what makes
this file trainer-aware rather than trainer-agnostic: a reward clamp is a property of a
specific implementation, so an assertion written for one trainer is noise when pointed at
another. The first version of this file asserted NeMo RL's schema against the superseded
Unsloth config and failed for the right reason in the wrong place.

Audited against NeMo RL `55296257c96d49cd95c7d77613cb0f36bd4a4dc7`, reading
`nemo_rl/algorithms/grpo.py` and `nemo_rl/algorithms/advantage_estimator.py` in
`../_vendor/nemo-rl/`. Findings:

    grpo.scale_rewards()            clamps to [source_min, source_max] BEFORE the linear
                                    remap; ships with source_min = 0.0. Gated by
                                    `reward_scaling.enabled`, default false.
    grpo._clip_grpo_advantages()    `advantage_clip_low/high`, both guarded by
                                    `is not None`, both default null.
    grpo._apply_message_level_...   `invalid_tool_call_advantage` and
                                    `malformed_thinking_advantage` OVERWRITE the advantage
                                    span of a flagged assistant message. Early-return when
                                    both are None; both default null.
    GRPOAdvantageEstimator          A = (r - baseline) / (std + 1e-6). Division by the
                                    group's standard deviation, not by a running maximum,
                                    so it is sign-agnostic and safe.

The stock defaults are safe. This file exists because *our* config is not the stock config,
and one `enabled: true` copied from an upstream example would reinstate exactly the hazard
ADR 0006 was written to prevent.

Three groups of checks:

  * provenance - every GRPO-shaped config in the repo is classified. A config for a trainer
    we did not choose must say so; an unrecognised schema fails rather than passing quietly.
  * NeMo RL config assertions - the knobs listed above, on configs we will actually run.
  * arithmetic characterisations - no config, no GPU, no model. They restate the upstream
    formulas on plain floats and show what each knob would do to the gate. Trainer-agnostic:
    (r - mu) / sigma is the same operation wherever it is implemented.

None of this replaces carry-forward item 4e - injecting a genuinely gated rollout into a
single-step run and asserting it receives the most negative advantage in its group. That
remains blocked on the ADR 0007 adapter.

If a check here fails, the fix is in the trainer's config or a thin adapter. It is never to
raise `GATE_REWARD`: that trades a detectable misconfiguration for an undetectable hack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from rewards.compose import (
    GATE_REWARD,
    REWARD_DID_NOTHING,
    REWARD_MIN,
    REWARD_SOLVED,
)

# --- the decision this file encodes -----------------------------------------------------

CHOSEN_TRAINER = "nemo_rl"
TRAINER_DECIDED_ON = "2026-08-03"

# The clone `analysis/versions.py` pins. Restated so a re-pin trips step 3 of the re-pinning
# procedure in `docs/pinned_versions.md` rather than leaving this file quietly stale.
AUDITED_NEMO_RL_COMMIT = "55296257c96d49cd95c7d77613cb0f36bd4a4dc7"

# `grpo.scale_rewards()` ships with this source interval. It is why `reward_scaling.enabled`
# is the most dangerous key in the trainer's config: the clamp happens before the remap, so
# a -1.0 never reaches the linear map at all.
UPSTREAM_DEFAULT_SOURCE_MIN = 0.0
UPSTREAM_DEFAULT_SOURCE_MAX = 1.0

# Keys that overwrite or truncate an advantage. Every one must stay null on our side.
ADVANTAGE_OVERRIDE_KEYS = (
    "advantage_clip_low",
    "advantage_clip_high",
    "invalid_tool_call_advantage",
    "malformed_thinking_advantage",
)

# Schema fingerprints. Deliberately keys that only one stack has, so a config that is
# neither is reported as neither instead of being coerced into the nearest match.
NEMO_RL_MARKERS = frozenset(
    {
        "adv_estimator",
        "use_leave_one_out_baseline",
        "reward_scaling",
        "num_prompts_per_step",
        "num_generations_per_prompt",
        "use_dynamic_sampling",
        "async_grpo",
    }
)
TRL_MARKERS = frozenset(
    {
        "warmup_ratio",
        "weight_decay",
        "per_device_train_batch_size",
        "num_generations",
        "scale_rewards",
        "loss_type",
        "max_completion_length",
        "importance_sampling_level",
        "mask_truncated_completions",
    }
)
NEMO_RL_TOP_LEVEL = frozenset({"policy", "cluster", "loss_fn", "checkpointing"})

# Directories with no trainer config that would otherwise dominate the scan: 36 task.yaml
# files, the vendored clones, virtualenvs.
SCAN_EXCLUDED_DIRS = frozenset({".git", ".venv", "venv", "_vendor", "__pycache__", "node_modules", "scratch"})
SCAN_EXCLUDED_PATHS = ("benchmark/tasks",)


# --- discovery ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """rewards/tests/this_file -> repo root."""
    return Path(__file__).resolve().parents[2]


def _is_scannable(path: Path, root: Path) -> bool:
    if any(part in SCAN_EXCLUDED_DIRS for part in path.parts):
        return False
    rel = path.relative_to(root).as_posix()
    return all(excluded not in rel for excluded in SCAN_EXCLUDED_PATHS)


def _load(path: Path) -> dict[str, Any] | None:
    """Parse a YAML file, tolerating anything that is not a plain mapping.

    OmegaConf interpolations (`${grpo.normalize_rewards}`) survive `safe_load` as strings,
    which is what we want: an interpolation is a deliberate alias, and asserting on the
    literal would be asserting on the wrong file.
    """
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, UnicodeDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _grpo_section(doc: dict[str, Any]) -> dict[str, Any] | None:
    """Return the `grpo` mapping, whether at the top level or one level down."""
    section = doc.get("grpo")
    if isinstance(section, dict):
        return section
    for value in doc.values():
        if isinstance(value, dict) and isinstance(value.get("grpo"), dict):
            return value["grpo"]
    return None


def _classify(doc: dict[str, Any], grpo: dict[str, Any]) -> str:
    """Which trainer's schema is this? Returns 'nemo_rl', 'trl', or 'unknown'.

    Scored rather than first-match: a config in transition may carry a few keys from both,
    and the honest answer is whichever it looks more like - or 'unknown' on a tie, which
    fails and asks a human.
    """
    keys = set(grpo)
    nemo_score = len(keys & NEMO_RL_MARKERS) + len(set(doc) & NEMO_RL_TOP_LEVEL)
    trl_score = len(keys & TRL_MARKERS)
    if nemo_score > trl_score:
        return "nemo_rl"
    if trl_score > nemo_score:
        return "trl"
    return "unknown"


def _status(doc: dict[str, Any]) -> str:
    value = doc.get("status")
    return value.strip().lower() if isinstance(value, str) else "active"


def _discover() -> list[dict[str, Any]]:
    root = _repo_root()
    found: list[dict[str, Any]] = []
    for pattern in ("*.yaml", "*.yml"):
        for path in sorted(root.rglob(pattern)):
            if not _is_scannable(path, root):
                continue
            doc = _load(path)
            if doc is None:
                continue
            grpo = _grpo_section(doc)
            if grpo is None:
                continue
            found.append(
                {
                    "path": path.relative_to(root),
                    "doc": doc,
                    "grpo": grpo,
                    "trainer": _classify(doc, grpo),
                    "status": _status(doc),
                }
            )
    return found


CONFIGS = _discover()
ACTIVE = [c for c in CONFIGS if c["status"] != "superseded"]
NEMO_RL_ACTIVE = [c for c in ACTIVE if c["trainer"] == "nemo_rl"]
SUPERSEDED = [c for c in CONFIGS if c["status"] == "superseded"]


def _ids(entries: list[dict[str, Any]]) -> list[str]:
    return [str(e["path"]) for e in entries]


_needs_nemo_rl = pytest.mark.skipif(
    not NEMO_RL_ACTIVE,
    reason="no active NeMo RL training config yet - see test_a_trainer_config_will_be_checked",
)


# --- provenance --------------------------------------------------------------------------


def test_a_trainer_config_will_be_checked():
    """A skipped suite must not be mistaken for a passing one.

    While no active NeMo RL config exists this records that fact. The moment one is
    committed the assertions below start running against it. If a config exists but the scan
    misses it - written as JSON, or living outside the repo - this is where that shows up,
    because it will still report zero and the guarantee will be silently absent.
    """
    if not NEMO_RL_ACTIVE:
        pytest.skip(
            f"No active {CHOSEN_TRAINER} config found. Scanned {len(CONFIGS)} GRPO-shaped "
            f"file(s): {', '.join(str(c['path']) for c in CONFIGS) or 'none'}. Expected "
            f"while the training config is still being written. When one lands, confirm it "
            f"appears here before trusting anything else in this file."
        )
    assert NEMO_RL_ACTIVE, "unreachable"


@pytest.mark.parametrize("entry", ACTIVE, ids=_ids(ACTIVE))
def test_every_active_config_targets_the_chosen_trainer(entry: dict[str, Any]):
    """Trainer selection is resolved; a config for anything else must say so.

    A config left lying around for a trainer we did not choose is worse than a deleted one:
    it will be read as current, and the hazards it does not guard against are different from
    the ones we audited. Mark it `status: superseded` with a `superseded_by` pointing at its
    replacement, which keeps its reasoning - the VRAM levers and the integration-risk notes
    survive a change of trainer - without letting it be mistaken for live configuration.
    """
    trainer = entry["trainer"]
    if trainer == CHOSEN_TRAINER:
        return

    if trainer == "unknown":
        pytest.fail(
            f"{entry['path']}: cannot tell which trainer this targets. It has a `grpo` "
            f"section but no decisive schema markers, so no hazard assertions apply and "
            f"none would have run. Either make the schema recognisable or mark it "
            f"`status: superseded`."
        )

    pytest.fail(
        f"{entry['path']}: targets `{trainer}`, but trainer selection resolved to "
        f"`{CHOSEN_TRAINER}` on {TRAINER_DECIDED_ON}. The reward-clamp audit in this file "
        f"was performed against NeMo RL {AUDITED_NEMO_RL_COMMIT[:8]} and does not transfer. "
        f"Add `status: superseded` and `superseded_by: <path>` to keep the file for its "
        f"reasoning, or delete it."
    )


@pytest.mark.parametrize("entry", SUPERSEDED, ids=_ids(SUPERSEDED))
def test_a_superseded_config_names_its_replacement(entry: dict[str, Any]):
    """`status: superseded` must point somewhere real.

    Otherwise the marker is just a way to switch off the checks. Naming the successor makes
    the decision traceable from either end, and breaks loudly if the successor is renamed.
    """
    successor = entry["doc"].get("superseded_by")
    assert isinstance(successor, str) and successor.strip(), (
        f"{entry['path']}: marked superseded but does not say by what. Add "
        f"`superseded_by: configs/training/<the-replacement>.yaml`."
    )
    target = _repo_root() / successor.strip()
    assert target.exists(), (
        f"{entry['path']}: `superseded_by: {successor}` does not exist. A superseded config "
        f"whose replacement is missing means the checks in this file are running against "
        f"nothing at all."
    )


# --- NeMo RL config assertions -----------------------------------------------------------


@_needs_nemo_rl
@pytest.mark.parametrize("entry", NEMO_RL_ACTIVE, ids=_ids(NEMO_RL_ACTIVE))
def test_reward_scaling_is_disabled(entry: dict[str, Any]):
    """`reward_scaling.enabled` must be false, or its source interval must admit -1.0.

    `scale_rewards()` clamps to [source_min, source_max] before remapping. With the upstream
    source_min of 0.0, every hard gate and every negative regression lands on the same value
    as doing nothing, and the ordering ADR 0006 depends on is gone.

    Upstream does print `[reward_scaling] WARNING: N rewards are outside the configured
    source range`, so the failure is loud in the logs - but only if someone reads them, and
    only after the run has started. This assertion is cheaper.
    """
    path, grpo = entry["path"], entry["grpo"]
    scaling = grpo.get("reward_scaling")
    if scaling is None:
        pytest.fail(
            f"{path}: `grpo.reward_scaling` is absent. `RewardScalingConfig.enabled` is a "
            f"required key upstream, so an absent section is a config that has not been "
            f"thought about. Set `enabled: false` explicitly."
        )

    if scaling.get("enabled") in (False, None):
        return

    source_min = float(scaling.get("source_min", UPSTREAM_DEFAULT_SOURCE_MIN))
    assert source_min <= REWARD_MIN, (
        f"{path}: reward_scaling is enabled with source_min={source_min}, which clamps "
        f"{REWARD_MIN} away. Every hard gate would collapse onto the value of doing nothing "
        f"and `tests_modified` would stop costing anything (ADR 0006). Either set "
        f"`enabled: false`, or set `source_min: {REWARD_MIN}`. Do not compensate by raising "
        f"GATE_REWARD."
    )


@_needs_nemo_rl
@pytest.mark.parametrize("entry", NEMO_RL_ACTIVE, ids=_ids(NEMO_RL_ACTIVE))
def test_advantage_overrides_are_null(entry: dict[str, Any]):
    """Nothing may clip or overwrite a computed advantage.

    `advantage_clip_low` truncates from below - at 0.0 it erases the gate. The two message
    level penalties are worse: they *overwrite* the advantage span of any assistant message
    the NeMo Gym detector flags. A coding agent is heavy on tool calls, so an episode gated
    for modifying `tests/**` could have its most negative advantage replaced by whatever
    constant is configured, for reasons unrelated to the gate.
    """
    path, grpo = entry["path"], entry["grpo"]
    for key in ADVANTAGE_OVERRIDE_KEYS:
        value = grpo.get(key)
        assert value is None, (
            f"{path}: `{key}` is {value!r}, but must stay null. Anything else overrides or "
            f"truncates advantages computed from the reward, decoupling the gate from what "
            f"the trainer optimises (ADR 0006)."
        )


@_needs_nemo_rl
@pytest.mark.parametrize("entry", NEMO_RL_ACTIVE, ids=_ids(NEMO_RL_ACTIVE))
def test_normalize_rewards_is_stated_not_inherited(entry: dict[str, Any]):
    """`normalize_rewards` must be an explicit bool.

    Either value is safe: the estimator divides by the group's standard deviation, not by a
    running maximum and not by a min-max span, so the operation is sign-agnostic and a
    negative reward stays negative. This asserts only that the key is present and boolean -
    a marker that the audit covered this knob, so a future reader does not have to re-derive
    whether normalisation is safe for a [-1, 1] domain.
    """
    path, grpo = entry["path"], entry["grpo"]
    value = grpo.get("normalize_rewards")
    assert isinstance(value, bool), (
        f"{path}: `normalize_rewards` is {value!r}. Expected an explicit bool. The "
        f"estimator's normalisation is `(r - baseline) / (std + 1e-6)`, which is safe for "
        f"[-1, 1]; state the intent rather than inheriting it."
    )


# --- arithmetic: what each knob would do to the gate -------------------------------------
#
# Restatements of the upstream formulas on plain floats. They assert ordering rather than
# exact values, because ordering is the invariant ADR 0006 needs and it survives any
# positive rescaling - which is all the leave-one-out baseline and the std division are.
# Trainer-agnostic by construction, so they keep their value if the trainer changes again.


def _grpo_advantages(rewards: list[float], *, leave_one_out: bool, normalize: bool) -> list[float]:
    """`GRPOAdvantageEstimator.compute_advantage`, for one prompt group.

    A = r - baseline, then optionally divided by the group std with an epsilon guard. The
    leave-one-out baseline for sample i is the mean of the others; both variants are affine
    and increasing in r, so neither can reorder a group.
    """
    n = len(rewards)
    total = sum(rewards)
    if leave_one_out and n > 1:
        baselines = [(total - r) / (n - 1) for r in rewards]
    else:
        baselines = [total / n] * n

    advantages = [r - b for r, b in zip(rewards, baselines)]
    if not normalize:
        return advantages

    mean = total / n
    std = (sum((r - mean) ** 2 for r in rewards) / n) ** 0.5
    if std <= 0:
        return advantages
    return [a / (std + 1e-6) for a in advantages]


def _scale_rewards(
    rewards: list[float], *, source_min: float, source_max: float, target_min: float, target_max: float
) -> list[float]:
    """`grpo.scale_rewards()`: clamp to the source interval, then remap linearly."""
    span = source_max - source_min
    clamped = [min(max(r, source_min), source_max) for r in rewards]
    return [target_min + (r - source_min) / span * (target_max - target_min) for r in clamped]


@pytest.mark.parametrize("leave_one_out", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_grpo_advantage_keeps_the_gate_at_the_bottom(leave_one_out: bool, normalize: bool):
    """A gated rollout must receive the most negative advantage in its group.

    This is the property carry-forward item 4e checks end to end. Here it is checked on the
    arithmetic alone, under every combination of the two estimator flags we might set.
    """
    group = [REWARD_SOLVED, 0.5, REWARD_DID_NOTHING, GATE_REWARD]
    advantages = _grpo_advantages(group, leave_one_out=leave_one_out, normalize=normalize)

    gate_advantage = advantages[-1]
    assert gate_advantage == min(advantages)
    assert gate_advantage < 0.0
    assert advantages[2] > gate_advantage, "doing nothing must beat being gated"
    assert all(a == a for a in advantages), "no NaN: std is non-zero for a mixed group"


def test_a_uniform_group_produces_no_signal_but_no_nan():
    """Every rollout gated: std is 0, the epsilon guard holds, advantages vanish.

    Upstream masks on `std > 0` and leaves those advantages undivided. The group carries no
    learning signal either way - correct, since nothing in it was better than anything else -
    but it must not poison the batch with NaN.
    """
    advantages = _grpo_advantages([GATE_REWARD] * 4, leave_one_out=True, normalize=True)
    assert all(a == a for a in advantages)
    assert all(a == 0.0 for a in advantages)


def test_reward_scaling_with_the_upstream_source_min_erases_the_gate():
    """Characterisation of the hazard. If this stops holding, upstream changed.

    With source_min = 0.0 the clamp maps GATE_REWARD onto the same value as doing nothing,
    before the advantage is computed. The distinction ADR 0006 pays for disappears without an
    exception, a failing test, or a difference in the loss.
    """
    group = [REWARD_SOLVED, REWARD_DID_NOTHING, GATE_REWARD]
    scaled = _scale_rewards(
        group,
        source_min=UPSTREAM_DEFAULT_SOURCE_MIN,
        source_max=UPSTREAM_DEFAULT_SOURCE_MAX,
        target_min=UPSTREAM_DEFAULT_SOURCE_MIN,
        target_max=UPSTREAM_DEFAULT_SOURCE_MAX,
    )
    assert scaled[-1] == scaled[-2], "the hazard: gated and did-nothing become the same reward"

    advantages = _grpo_advantages(scaled, leave_one_out=True, normalize=True)
    assert advantages[-1] == advantages[-2], "and the same advantage, so the gate is free"


def test_reward_scaling_is_harmless_when_the_source_interval_admits_the_gate():
    """The supported way to enable it: widen the source interval to the real domain.

    Recorded so that "disable it" is not the only remembered option.
    """
    group = [REWARD_SOLVED, REWARD_DID_NOTHING, GATE_REWARD]
    scaled = _scale_rewards(
        group, source_min=REWARD_MIN, source_max=REWARD_SOLVED, target_min=REWARD_MIN, target_max=REWARD_SOLVED
    )
    assert scaled == group

    advantages = _grpo_advantages(scaled, leave_one_out=True, normalize=True)
    assert advantages[-1] == min(advantages)


def test_clipping_advantages_at_zero_would_hide_the_gate():
    """Why `advantage_clip_low` must stay null rather than be set to a round number.

    Any lower bound at or above zero collapses the gated rollout onto the rollouts that
    merely underperformed - the same failure as the reward clamp, one layer down.
    """
    group = [REWARD_SOLVED, REWARD_DID_NOTHING, GATE_REWARD]
    advantages = _grpo_advantages(group, leave_one_out=True, normalize=True)

    clipped = [max(a, 0.0) for a in advantages]
    assert clipped[-1] == clipped[-2] == 0.0
    assert advantages[-1] < advantages[-2], "unclipped, the ordering was intact"


def test_the_gate_is_the_bottom_of_the_declared_scale():
    """Guard the assumption every test above rests on.

    `rewards/tests/test_compose.py` owns this invariant; it is restated here because the
    arithmetic in this file is meaningless if GATE_REWARD is not the minimum.
    """
    assert GATE_REWARD == REWARD_MIN
    assert REWARD_MIN < REWARD_DID_NOTHING < REWARD_SOLVED
