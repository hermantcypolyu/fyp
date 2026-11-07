"""Phase 6I closure - manifest executability is a gate, not a printed warning.

`run_manifest.py` used to *report* that a manifest described an unstartable run and then
write it out with exit code 0 anyway. A reader who skimmed the output got a file on disk and
an exit status that both said "fine", and the Phase 6I preflight failed on a config nobody
had been stopped from writing. These tests pin the enforcement that replaced that, and the
single artefact-mapping table that replaced the three hand-copied output aliases.

Each numbered case corresponds to one named obligation:

  1-2   a runnable config exits 0, an unstartable one exits 2 and writes nothing
  3-5   the executed identity is required: tasks, rollouts_per_task, model name/revision
  6-8   each required collector output is required, keyed on the *output* name
  9-12  a rollout-only diagnostic run may not claim to hold out, optimise, or checkpoint
  13-14 the overlay's placeholders never substitute for the executed settings
  15-17 where the run writes is derived from one table, and pending paths stay null
  18-19 not-applicable fields and failed preflights are labelled for what they are
  20    the recorded Phase 6I evidence is byte-identical to what it was

CPU-only: no GPU, vLLM, model weights, network, NeMo or Ray. CLI tests write into `tmp_path`
so no run directory is created under `results/`.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_manifest.py"

#: The two Phase 6I directories whose contents are evidence and must not drift.
PHASE6I_RUNS = {
    "results/raw/phase6i_bugfix_offbyone_pagination_20261106T161132Z": (
        "cf4e1624402702d7ac766dcbcbb929e38661593a8d84b484bfc42d808669e273"
    ),
    "results/raw/phase6i_bugfix_offbyone_pagination_20261106T162531Z": (
        "beaf9c6c9e3844f903f71d6cc780fd3eeb336e8ddefab3670d0e19ec5a69396c"
    ),
}


def _load_module() -> Any:
    """Import scripts/run_manifest.py, which is a script rather than a package module."""
    spec = importlib.util.spec_from_file_location("run_manifest_enforcement_under_test", SCRIPT)
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
        "run_id": "phase6i_enforcement_probe",
        "split": "train",
        "seed": 42,
        "input_checkpoint": None,
        "output_checkpoint": None,
        "note": None,
    }
    kwargs.update(overrides)
    return rm.build_manifest(**kwargs)


def _config(tmp: Path, *, name: str = "merged_config.yaml", **sections: Any) -> str:
    """Write a materialised eval config under `tmp` and return its absolute path.

    Defaults describe a legitimate rollout-only diagnostic run; `sections` shallow-overrides
    the top-level YAML keys so a test can break exactly one thing.

    The declared `output.*` paths stay repo-relative - that is where a real run writes - but
    the config file itself lives in `tmp_path`, so nothing is created under `results/`.
    """
    stem = f"results/raw/{tmp.name}"
    document: dict[str, Any] = {
        "model": {"name": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "revision": "abc123", "adapter": None},
        "sampling": {"temperature": 1.0, "top_p": 1.0, "seed": 42, "max_tokens_per_turn": 1024},
        "episode": {"max_turns": 30, "max_tool_calls": 40, "timeout_s_per_rollout": 900},
        "run": {
            "task_subset": ["t1", "t2"],
            "rollouts_per_task": 4,
            "min_valid_rollouts": 4,
            "min_distinct_rewards": 2,
            "diagnostic_only": True,
        },
        "output": {
            "run_id": "phase6i_enforcement_probe",
            "rollouts": f"{stem}/rollouts.jsonl",
            "private": f"{stem}/rollouts_private.jsonl",
            "failures": f"{stem}/failures.jsonl",
        },
        "tasks": {"split": "train", "heldout_enabled": False},
        "reward": {"live": "reward_v1", "analysed": "reward_v2", "reward_v2_computed_inline": False},
        "server": {"cuda_visible_devices": "0", "served_model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
        "context": {"max_model_len": 8192, "reserved_completion_tokens": 1024, "safety_margin_tokens": 64},
    }
    for key, value in sections.items():
        if isinstance(value, dict) and isinstance(document.get(key), dict):
            document[key] = {**document[key], **value}
        else:
            document[key] = value
    target = tmp / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return str(target)


def _rewrite(rel: str, edit: Any) -> str:
    """Apply `edit(document)` to a written config in place and return its path."""
    path = Path(rel)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    edit(document)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return rel


def _report(rm: Any, rel: str | None, **overrides: Any) -> dict[str, Any]:
    return rm.executable_report(_base(rm, eval_config=rel, **overrides))


def _blockers(report: dict[str, Any]) -> str:
    return "\n".join(report["blockers"])


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


# --- case 1-2: the exit-code contract ------------------------------------------------


def test_01_a_valid_materialised_diagnostic_config_is_executable(rm: Any, tmp_path: Any) -> None:
    """The gate has to be passable, or it is just a wall."""
    report = _report(rm, _config(tmp_path))

    assert report["ok"] is True, _blockers(report)
    assert report["status"] == "executable"
    assert report["blockers"] == []


def test_02_a_non_materialised_overlay_is_refused_and_writes_nothing(rm: Any, tmp_path: Any) -> None:
    """An overlay names no task list, so it cannot describe a run; the CLI must not write it."""
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("extends: configs/training/pilot_rollouts_1p5b.yaml\n", encoding="utf-8")
    target = tmp_path / "out"

    result = _run_cli("--arm", "pilot", "--run-id", "refused", "--out", str(target), "--eval-config", str(overlay))

    assert result.returncode == 2, result.stdout + result.stderr
    assert "executed.materialised is false" in result.stderr
    assert "nothing was written to" in result.stderr
    assert not target.exists(), "a failed preflight must leave no manifest behind"


def test_02b_a_manifest_with_no_execution_claim_still_exits_zero(rm: Any, tmp_path: Any) -> None:
    """`--arm baseline` claims nothing to execute, so there is nothing to refuse."""
    target = tmp_path / "out"

    result = _run_cli("--arm", "baseline", "--run-id", "no_claim", "--out", str(target))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "no_execution_claim" in result.stdout
    assert (target / "manifest.json").is_file()


# --- case 3-5: the executed identity ---------------------------------------------------


def test_03_a_null_executed_task_list_is_a_blocker(rm: Any, tmp_path: Any) -> None:
    report = _report(rm, _config(tmp_path, run={"task_subset": None}))

    assert report["ok"] is False
    assert "no executed task" in _blockers(report)


def test_03b_an_empty_executed_task_list_is_a_blocker(rm: Any, tmp_path: Any) -> None:
    """`[]` and `null` are the same claim: nothing to run."""
    report = _report(rm, _config(tmp_path, run={"task_subset": []}))

    assert report["ok"] is False
    assert "no executed task" in _blockers(report)


def test_04_a_missing_rollouts_per_task_is_a_blocker(rm: Any, tmp_path: Any) -> None:
    rel = _rewrite(_config(tmp_path), lambda document: document["run"].pop("rollouts_per_task"))

    report = _report(rm, rel)

    assert report["ok"] is False
    assert "rollouts_per_task" in _blockers(report)


@pytest.mark.parametrize("missing", ["name", "revision"])
def test_05_incomplete_model_identity_is_a_blocker(rm: Any, tmp_path: Any, missing: str) -> None:
    """A run whose weights cannot be identified cannot be reproduced."""
    model = {"name": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "revision": "abc123"}
    model[missing] = None

    report = _report(rm, _config(tmp_path, model=model))

    assert report["ok"] is False
    assert "model identity is incomplete" in _blockers(report)


# --- case 6-8: required collector outputs, keyed on the output name --------------------


@pytest.mark.parametrize(
    ("output_key", "artifact_field"),
    [("rollouts", "trace"), ("private", "private_trace"), ("failures", "failures")],
)
def test_06_08_a_missing_required_output_is_a_blocker(
    rm: Any, tmp_path: Any, output_key: str, artifact_field: str
) -> None:
    """The blocker names the *output* key but points at the artefact field it feeds.

    That pairing is the whole reason the mapping is a table: when `output.private` was
    checked against `artifacts.private` while the value lived in `artifacts.private_trace`,
    a correct config was reported as broken.
    """
    rel = _rewrite(_config(tmp_path), lambda document: document["output"].__setitem__(output_key, None))

    report = _report(rm, rel)

    assert report["ok"] is False
    assert f"output.{output_key}" in _blockers(report)
    assert f"artifacts.{artifact_field}" in _blockers(report)


# --- case 9-12: a rollout-only diagnostic run may not claim to do more -----------------


def test_09_heldout_enabled_is_a_blocker(rm: Any, tmp_path: Any) -> None:
    """The held-out split is sealed; a config that unseals it must not run."""
    report = _report(rm, _config(tmp_path, tasks={"heldout_enabled": True}))

    assert report["ok"] is False
    assert "heldout_enabled must be false" in _blockers(report)


def test_10_optimizer_enabled_is_a_blocker(rm: Any, tmp_path: Any) -> None:
    report = _report(rm, _config(tmp_path, run={"optimizer_enabled": True}))

    assert report["ok"] is False
    assert "optimizer_enabled is true" in _blockers(report)


def test_11_positive_training_iterations_is_a_blocker(rm: Any, tmp_path: Any) -> None:
    report = _report(rm, _config(tmp_path, run={"training_iterations": 3}))

    assert report["ok"] is False
    assert "training_iterations is 3" in _blockers(report)


def test_11b_zero_training_iterations_does_not_block(rm: Any, tmp_path: Any) -> None:
    """The boundary: 0 updates is what a diagnostic run performs."""
    report = _report(rm, _config(tmp_path, run={"training_iterations": 0}))

    assert report["ok"] is True, _blockers(report)


def test_12_expect_output_checkpoint_is_a_blocker(rm: Any, tmp_path: Any) -> None:
    report = _report(rm, _config(tmp_path, run={"expect_output_checkpoint": True}))

    assert report["ok"] is False
    assert "expect_output_checkpoint is true" in _blockers(report)


# --- case 13-14: the overlay never substitutes for the executed settings ---------------


def test_13_the_overlay_null_task_list_does_not_block_when_executed_tasks_exist(rm: Any, tmp_path: Any) -> None:
    """`pilot.values.tasks` is null by design; only `executed.settings.tasks` may gate."""
    manifest = _base(rm, eval_config=_config(tmp_path))

    assert manifest["pilot"]["values"]["tasks"] is None
    assert manifest["executed"]["settings"]["tasks"] == ["t1", "t2"]
    assert "pilot.tasks (not yet selected)" not in rm.executable_report(manifest)["blockers"]


def test_14_the_overlay_rollouts_do_not_override_the_executed_rollouts(rm: Any, tmp_path: Any) -> None:
    """The overlay says 8, the materialised config says 4, and the run uses 4."""
    manifest = _base(rm, eval_config=_config(tmp_path))

    assert manifest["pilot"]["values"]["rollouts_per_task"] == 8
    assert manifest["executed"]["settings"]["rollouts_per_task"] == 4
    assert "rollouts_per_task" not in _blockers(rm.executable_report(manifest))


# --- case 15-17: one table decides where the run writes --------------------------------


def test_15_run_dir_and_trace_are_derived_from_the_same_declaration(rm: Any, tmp_path: Any) -> None:
    """`artifacts.run_dir` is the parent of the rollouts file, not a second opinion."""
    rel = _config(tmp_path)
    manifest = _base(rm, eval_config=rel)
    artifacts = manifest["artifacts"]
    declared = yaml.safe_load(Path(rel).read_text(encoding="utf-8"))["output"]["rollouts"]

    assert artifacts["trace"] == declared
    assert artifacts["run_dir"] == str(Path(declared).parent)
    assert artifacts["summary"] == str(Path(declared).parent / "collection_summary.json")


def test_16_private_and_failure_outputs_map_to_their_own_artifact_fields(rm: Any, tmp_path: Any) -> None:
    """The three collector outputs are distinct; the aliasing bug was these being confused."""
    rel = _config(tmp_path)
    manifest = _base(rm, eval_config=rel)
    declared = yaml.safe_load(Path(rel).read_text(encoding="utf-8"))["output"]

    assert manifest["artifacts"]["trace"] == declared["rollouts"]
    assert manifest["artifacts"]["private_trace"] == declared["private"]
    assert manifest["artifacts"]["failures"] == declared["failures"]

    mapped = rm.map_outputs(declared)
    assert mapped["trace"] == declared["rollouts"]
    assert mapped["private_trace"] == declared["private"]
    assert mapped["failures"] == declared["failures"]
    assert len({mapped["trace"], mapped["private_trace"], mapped["failures"]}) == 3


def test_17_log_and_server_pid_may_stay_null_before_startup(rm: Any, tmp_path: Any) -> None:
    """These arrive when the server starts, so their absence is a warning, not a blocker."""
    manifest = _base(rm, eval_config=_config(tmp_path))
    report = rm.executable_report(manifest)

    assert manifest["artifacts"]["log"] is None
    assert manifest["artifacts"]["server_pid"] is None
    assert report["ok"] is True, _blockers(report)
    warned = {warning["field"] for warning in report["warnings"]}
    assert {"artifacts.log", "artifacts.server_pid"} <= warned


# --- case 18-19: labels a reader can act on --------------------------------------------


def test_18_a_null_kl_coefficient_is_not_applicable_rather_than_blocking(rm: Any, tmp_path: Any) -> None:
    """No reference-policy KL term exists here, so the field can never be filled in.

    Reported as a blocker it would read like a decision someone had failed to make.
    """
    manifest = _base(rm, eval_config=_config(tmp_path))
    report = rm.executable_report(manifest)

    assert manifest["protocol"]["grpo"]["kl_coefficient"] is None
    assert "protocol.grpo.kl_coefficient" in manifest["unresolved"]
    assert not any("kl_coefficient" in blocker for blocker in report["blockers"])
    assert any("kl_coefficient" in item["field"] for item in report["not_applicable"])


def test_18b_a_dirty_tree_is_a_warning_not_a_blocker(rm: Any, tmp_path: Any) -> None:
    """A dirty tree is a fact about the working copy that no config edit can fix."""
    manifest = _base(rm, eval_config=_config(tmp_path))
    manifest["git"]["dirty"] = True
    report = rm.classify_fields(manifest)

    assert not any("git.dirty" in blocker for blocker in report["blockers"])
    assert any("git.dirty" in warning["field"] for warning in report["warnings"])


def test_19_a_failed_preflight_is_never_labelled_planned(rm: Any, tmp_path: Any) -> None:
    """`outcome.status` says the manifest was written, not that the run may start."""
    bad = _report(rm, _config(tmp_path, run={"task_subset": None}))
    good = _report(rm, _config(tmp_path))
    manifest = _base(rm, eval_config=_config(tmp_path))

    assert bad["status"] == "preflight_failed"
    assert bad["ok"] is False
    assert good["status"] == "executable"
    assert manifest["outcome"]["status"] == "planned"
    assert manifest["outcome"]["failure_reason"] is None


# --- case 20: the evidence does not move ------------------------------------------------


@pytest.mark.parametrize(("rel", "expected"), sorted(PHASE6I_RUNS.items()))
def test_20_the_phase6i_failed_preflight_manifest_is_byte_identical(rel: str, expected: str) -> None:
    """Both Phase 6I directories record what actually happened; they must not be rewritten."""
    manifest = REPO_ROOT / rel / "manifest.json"
    if not manifest.is_file():
        pytest.skip(f"{rel} is not present in this checkout")

    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == expected


def test_20b_the_mapping_table_is_internally_consistent(rm: Any) -> None:
    """The table validates itself at import, and every artefact field it names is known."""
    rm._check_output_map()

    assert set(rm.EXECUTED_ARTIFACT_FIELDS) <= rm.KNOWN_ARTIFACT_FIELDS
    assert rm.EXECUTED_REQUIRED_OUTPUTS == ("rollouts", "private", "failures")
    assert rm.EXECUTED_OUTPUT_ALIASES == {
        "rollouts": "trace",
        "private": "private_trace",
        "failures": "failures",
    }


def test_20c_the_cli_json_output_is_valid_json(rm: Any, tmp_path: Any) -> None:
    """`--json` is the machine-readable view, so it has to parse."""
    target = tmp_path / "out"

    result = _run_cli("--arm", "baseline", "--run-id", "json_probe", "--out", str(target), "--json")

    assert result.returncode == 0, result.stderr
    payload = result.stdout[result.stdout.index("{") :]
    document = json.loads(payload[: payload.rindex("}") + 1])
    assert document["outcome"]["status"] == "planned"
    assert json.dumps(document)  # round-trips, so nothing unserialisable leaked in


def test_20d_a_second_write_without_force_exits_one(rm: Any, tmp_path: Any) -> None:
    """An existing differing manifest is a refusal (1), distinct from an unstartable run (2)."""
    target = tmp_path / "out"
    first = _run_cli("--arm", "baseline", "--run-id", "one", "--out", str(target))
    assert first.returncode == 0, first.stderr

    second = _run_cli("--arm", "baseline", "--run-id", "two", "--out", str(target))

    assert second.returncode == 1, second.stdout + second.stderr
    assert "refusing" in second.stderr
