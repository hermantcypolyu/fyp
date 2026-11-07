"""Phase 6I - a manifest must not be able to describe an unstartable run without saying so.

The Phase 6I preflight failed on a gate that could not be satisfied: the manifest hard-coded
`artifacts.*` to null, so a materialised run config produced a manifest that named no run
directory, no trace and no summary - a manifest that could not be told apart from one whose
run never happened. Three separate statements were missing, and each is pinned here:

  * *where* the run writes (``artifacts`` populated at creation, not after the fact);
  * *what* the run executes (``executed.settings`` authoritative over overlay placeholders);
  * *whether* it can start at all (``executable_report``, and the absence of the overlay
    placeholder from ``unresolved`` once a real task list exists).

These tests are CPU-only. They write temporary configs under ``results/`` (gitignored) and
delete them, so nothing on disk outside a temp dir is touched and no GPU is involved.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module() -> Any:
    """Import scripts/run_manifest.py, which is a script rather than a package module."""
    path = REPO_ROOT / "scripts" / "run_manifest.py"
    spec = importlib.util.spec_from_file_location("run_manifest_preflight_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rm() -> Any:
    return _load_module()


def _base(rm: Any, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "arm": "pilot",
        "run_id": "phase6i_unit_probe",
        "split": "train",
        "seed": 42,
        "input_checkpoint": None,
        "output_checkpoint": None,
        "note": None,
    }
    kwargs.update(overrides)
    return rm.build_manifest(**kwargs)


def _materialised_config(rel: str, *, run_id: str, tasks: list[str], rollouts: int = 4) -> str:
    """Write a materialised (no ``extends:``) eval config under ``results/`` and return its path.

    The three ``output.*`` paths are declared because the collector opens all three by name;
    a config without them is not runnable and the manifest is expected to say so.
    """
    target = REPO_ROOT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    stem = rel.rsplit("/", 1)[0]
    document = {
        "model": {"name": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "revision": "abc123", "adapter": None},
        "sampling": {"temperature": 1.0, "top_p": 1.0, "seed": 42, "max_tokens_per_turn": 1024},
        "episode": {"max_turns": 30, "max_tool_calls": 40, "timeout_s_per_rollout": 900},
        "run": {
            "task_subset": tasks,
            "rollouts_per_task": rollouts,
            "min_valid_rollouts": rollouts,
            "min_distinct_rewards": 2,
            "diagnostic_only": True,
        },
        "output": {
            "run_id": run_id,
            "rollouts": f"{stem}/rollouts.jsonl",
            "private": f"{stem}/rollouts_private.jsonl",
            "failures": f"{stem}/failures.jsonl",
        },
        "tasks": {"split": "train", "heldout_enabled": False},
        "reward": {"live": "reward_v1", "analysed": "reward_v2", "reward_v2_computed_inline": False},
        "server": {"cuda_visible_devices": "0", "served_model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
        "context": {"max_model_len": 8192, "reserved_completion_tokens": 1024, "safety_margin_tokens": 64},
    }
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return rel


def _cleanup(rel: str) -> None:
    target = REPO_ROOT / rel
    target.unlink(missing_ok=True)
    parent = target.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


# --- fix 1 & 2: overlay values are labelled as overlay values --------------------------


def test_pilot_values_are_nested_and_labelled_as_overlay_values(rm: Any) -> None:
    """`pilot.values` mirrors the overlay; it is not the run's executed identity."""
    pilot = _base(rm)["pilot"]

    assert pilot["values_from"] == "overlay_file"
    assert "executed.settings" in pilot["values_note"]
    assert set(pilot["values"]) >= {"tasks", "rollouts_per_task", "concurrency"}


def test_the_overlay_placeholders_are_recorded_verbatim_and_not_invented(rm: Any) -> None:
    """`tasks: null` and `rollouts_per_task: 8` are the committed file's own values."""
    values = _base(rm)["pilot"]["values"]

    assert values["tasks"] is None, "the overlay ships a null task list on purpose"
    assert values["rollouts_per_task"] == 8, "not the 4 the executed config will use"


# --- fix 3 & 9: executed.settings is authoritative, and it resolves the blocker ---------


def test_executed_settings_name_the_run_identity_not_the_overlay_placeholder(rm: Any) -> None:
    """The executed block is where a reader must look to learn what actually ran."""
    rel = "results/raw/phase6i_unit_probe_a/merged_config.yaml"
    try:
        manifest = _base(rm, eval_config=_materialised_config(rel, run_id="phase6i_unit_probe_a", tasks=["t1"]))
    finally:
        _cleanup(rel)

    assert manifest["executed"]["settings"]["tasks"] == ["t1"]
    assert manifest["executed"]["settings"]["rollouts_per_task"] == 4
    assert manifest["pilot"]["values"]["tasks"] is None, "the overlay value is unchanged"


def test_a_materialised_task_list_suppresses_the_overlay_placeholder_blocker(rm: Any) -> None:
    """Fix 9: the placeholder must stop blocking once a real task list is recorded."""
    rel = "results/raw/phase6i_unit_probe_b/merged_config.yaml"
    try:
        with_executed = _base(rm, eval_config=_materialised_config(rel, run_id="phase6i_unit_probe_b", tasks=["t1"]))
    finally:
        _cleanup(rel)
    without_executed = _base(rm)

    assert "pilot.tasks (not yet selected)" in without_executed["unresolved"]
    assert "pilot.tasks (not yet selected)" not in with_executed["unresolved"]


def test_the_placeholder_blocker_returns_when_no_real_task_list_is_recorded(rm: Any) -> None:
    """Suppression must be earned by a real value, not by the presence of an executed block."""
    rel = "results/raw/phase6i_unit_probe_c/merged_config.yaml"
    try:
        target = REPO_ROOT / rel
        _materialised_config(rel, run_id="phase6i_unit_probe_c", tasks=["t1"])
        document = yaml.safe_load(target.read_text(encoding="utf-8"))
        document["run"]["task_subset"] = None
        target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        manifest = _base(rm, eval_config=rel)
    finally:
        _cleanup(rel)

    assert (manifest["executed"]["settings"] or {}).get("tasks") is None
    assert "pilot.tasks (not yet selected)" in manifest["unresolved"]


# --- fix 4, 5, 6, 7: artifacts are populated at creation --------------------------------


def test_artifacts_name_the_run_directory_and_trace_before_anything_runs(rm: Any) -> None:
    """Fix 4 & 5: a planned manifest names its intended outputs instead of four nulls."""
    rel = "results/raw/phase6i_unit_probe_d/merged_config.yaml"
    try:
        artifacts = _base(rm, eval_config=_materialised_config(rel, run_id="phase6i_unit_probe_d", tasks=["t1"]))[
            "artifacts"
        ]
    finally:
        _cleanup(rel)

    assert artifacts["run_dir"] == "results/raw/phase6i_unit_probe_d"
    assert artifacts["trace"] == "results/raw/phase6i_unit_probe_d/rollouts.jsonl"


def test_artifacts_record_the_private_trace_failures_and_summary(rm: Any) -> None:
    """Fix 6: the private trace, failure log and summary are all named up front."""
    rel = "results/raw/phase6i_unit_probe_e/merged_config.yaml"
    try:
        artifacts = _base(rm, eval_config=_materialised_config(rel, run_id="phase6i_unit_probe_e", tasks=["t1"]))[
            "artifacts"
        ]
    finally:
        _cleanup(rel)

    assert artifacts["private_trace"] == "results/raw/phase6i_unit_probe_e/rollouts_private.jsonl"
    assert artifacts["failures"] == "results/raw/phase6i_unit_probe_e/failures.jsonl"
    # `output.summary` is not a config key; the collector writes this name beside the trace.
    assert artifacts["summary"] == "results/raw/phase6i_unit_probe_e/collection_summary.json"


def test_log_and_server_pid_may_stay_null_before_startup(rm: Any) -> None:
    """Fix 7: these two are produced by the server, so a pre-run manifest cannot know them."""
    rel = "results/raw/phase6i_unit_probe_f/merged_config.yaml"
    try:
        manifest = _base(rm, eval_config=_materialised_config(rel, run_id="phase6i_unit_probe_f", tasks=["t1"]))
    finally:
        _cleanup(rel)

    assert manifest["artifacts"]["log"] is None
    assert manifest["artifacts"]["server_pid"] is None
    # ...and their nullness does not make the manifest unrunnable.
    assert rm.executable_report(manifest)["ok"] is True


def test_artifacts_fall_back_to_the_directory_the_manifest_is_written_into(rm: Any) -> None:
    """Without an executed config there is nothing to declare - but the manifest still knows
    which directory it was written to, so `run_dir` is never a bare null."""
    artifacts = _base(rm, run_id="phase6i_unit_probe_g")["artifacts"]

    assert artifacts["run_dir"] == "results/raw/phase6i_unit_probe_g"
    assert artifacts["trace"] is None


def test_the_out_dir_argument_wins_over_the_run_id_derived_path(rm: Any) -> None:
    """A manifest written to `--out` must name that directory, not the default one."""
    manifest = _base(rm, out_dir=REPO_ROOT / "results" / "raw" / "elsewhere_6i")

    assert manifest["artifacts"]["run_dir"] == "results/raw/elsewhere_6i"


# --- fix 8 & the executability gate -----------------------------------------------------


def test_a_materialised_config_missing_an_output_path_is_not_materialised(rm: Any) -> None:
    """Fix 8: the collector opens `output.private` by name, so its absence is disqualifying."""
    rel = "results/raw/phase6i_unit_probe_h/merged_config.yaml"
    try:
        target = REPO_ROOT / rel
        _materialised_config(rel, run_id="phase6i_unit_probe_h", tasks=["t1"])
        document = yaml.safe_load(target.read_text(encoding="utf-8"))
        del document["output"]["private"]
        target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        executed = rm.executed_config(rel)
    finally:
        _cleanup(rel)

    assert executed["materialised"] is False
    assert "output.private" in executed["missing"]


def test_a_materialised_config_with_a_null_setting_is_not_materialised(rm: Any) -> None:
    """Fix 8: `materialised` must not be true while a setting the run needs is null."""
    rel = "results/raw/phase6i_unit_probe_i/merged_config.yaml"
    try:
        target = REPO_ROOT / rel
        _materialised_config(rel, run_id="phase6i_unit_probe_i", tasks=["t1"])
        document = yaml.safe_load(target.read_text(encoding="utf-8"))
        document["run"]["task_subset"] = None
        target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        executed = rm.executed_config(rel)
    finally:
        _cleanup(rel)

    assert executed["materialised"] is True, "every key is present..."
    assert executed["settings_complete"] is False, "...but the task list is empty"
    assert "settings are incomplete" in executed["error"]


def test_a_missing_config_is_not_executable(rm: Any) -> None:
    """A path that does not exist must not yield a manifest that looks startable."""
    manifest = _base(rm, eval_config="results/raw/does_not_exist_6i/merged_config.yaml")

    report = rm.executable_report(manifest)
    assert report["ok"] is False
    assert any("executed.materialised is false" in b for b in report["blockers"])


def test_an_overlay_is_not_executable(rm: Any) -> None:
    """An overlay resolves no `extends:`, so nothing ever expands it into a runnable file."""
    manifest = _base(rm, eval_config="configs/eval/phase6g_diagnostic_smoke.yaml")

    report = rm.executable_report(manifest)
    assert report["ok"] is False, "an overlay must never pass preflight"
    assert report["reason"]


def test_a_complete_materialised_manifest_is_executable(rm: Any) -> None:
    """The gate has to be passable, or it is just a wall."""
    rel = "results/raw/phase6i_unit_probe_j/merged_config.yaml"
    try:
        manifest = _base(rm, eval_config=_materialised_config(rel, run_id="phase6i_unit_probe_j", tasks=["t1"]))
    finally:
        _cleanup(rel)

    report = rm.executable_report(manifest)
    assert report["ok"] is True, report["reason"]
    assert report["blockers"] == []


def test_a_manifest_that_claims_nothing_about_execution_is_not_flagged(rm: Any) -> None:
    """A training arm passes no `--eval-config`; that is honest, not a failing gate."""
    report = rm.executable_report(_base(rm, arm="training"))

    assert report["ok"] is True
    assert report["reason"] is None
