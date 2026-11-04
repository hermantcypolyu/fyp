"""Phase 5.2 - the run manifest refuses to let a result be unattributable.

The manifest is the only artefact that says *what produced a number*. Two failure modes
matter, and neither raises on its own:

  * a manifest that quietly overwrites an earlier one, so the run whose settings changed
    leaves no trace of having used the old settings;
  * a manifest that records a plausible value for something it does not know - an unpinned
    model revision, a top_p nobody chose - so a later reader treats a guess as a fact.

Both are tested here against a temporary repository rather than by inspecting the real
script's constants, because the behaviour that matters is what it writes to disk.

These tests are CPU-only and do not need a GPU, torch or the NeMo stack. On this machine
`gpu.present` is legitimately false; the tests assert the *shape* of the record, not that
hardware exists.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module() -> Any:
    """Import scripts/run_manifest.py, which is a script rather than a package module."""
    path = REPO_ROOT / "scripts" / "run_manifest.py"
    spec = importlib.util.spec_from_file_location("run_manifest_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rm() -> Any:
    return _load_module()


def _base(rm: Any, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "arm": "baseline",
        "run_id": "unit_test_run",
        "split": "train",
        "seed": 42,
        "input_checkpoint": None,
        "output_checkpoint": None,
        "note": None,
    }
    kwargs.update(overrides)
    return rm.build_manifest(**kwargs)


# --- identity ------------------------------------------------------------------------


def test_manifest_has_every_field_phase_5_2_requires(rm: Any) -> None:
    """5.2 lists the manifest contents; each must be present under some name.

    The mapping is asserted explicitly so that a field rename cannot silently drop one:
    `run_id`, date, git commit, model, dataset version, reward version, trajectory schema
    version, training config hash, evaluation config hash, input/output checkpoint, GPU,
    seed, result folder.
    """
    manifest = _base(rm)
    required = {
        "run_id": manifest["run_id"],
        "date": manifest["written_at"],
        "git_commit": manifest["git"]["commit"],
        "model": manifest["protocol"]["model"]["name"],
        "dataset_version": manifest["dataset"]["version"]["sha256"],
        "reward_version": manifest["reward"]["live"],
        "trajectory_schema_version": manifest["trajectory_schema"]["sha256"],
        "training_config_hash": manifest["arm_config"]["sha256"],
        "evaluation_config_hash": manifest["protocol"]["file"]["sha256"],
        "input_checkpoint": manifest["checkpoints"]["input"],
        "output_checkpoint": manifest["checkpoints"]["output"],
        "gpu": manifest["gpu"],
        "seed": manifest["seed"],
        "result_folder": manifest["artifacts"]["run_dir"],
    }
    assert set(required) == {
        "run_id",
        "date",
        "git_commit",
        "model",
        "dataset_version",
        "reward_version",
        "trajectory_schema_version",
        "training_config_hash",
        "evaluation_config_hash",
        "input_checkpoint",
        "output_checkpoint",
        "gpu",
        "seed",
        "result_folder",
    }
    # The identity fields must be resolved now, not deferred.
    for name in (
        "run_id",
        "date",
        "git_commit",
        "model",
        "dataset_version",
        "reward_version",
        "trajectory_schema_version",
        "training_config_hash",
        "evaluation_config_hash",
        "seed",
    ):
        assert required[name] is not None, f"{name} was not resolved"


def test_both_arms_resolve_to_a_real_config_file(rm: Any) -> None:
    """A manifest for a nonexistent arm config would carry a null hash and look complete."""
    for arm in rm.ARM_CONFIGS:
        manifest = _base(rm, arm=arm)
        assert manifest["arm_config"]["present"] is True
        assert manifest["arm_config"]["sha256"] is not None


def test_unknown_arm_is_rejected_rather_than_guessed(rm: Any) -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        _base(rm, arm="training_v3")


# --- hashing ---------------------------------------------------------------------------


def test_config_hashes_change_when_the_config_changes(rm: Any, tmp_path: Path, monkeypatch: Any) -> None:
    """The whole point of a hash: an edited protocol must produce a different value.

    Without this, two runs could both claim `protocol_1p5b` while using different settings.
    """
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("meta:\n  name: p\n", encoding="utf-8")
    first = rm.sha256_of(protocol)
    protocol.write_text("meta:\n  name: p\ntraining:\n  learning_rate: 2.0e-5\n", encoding="utf-8")
    second = rm.sha256_of(protocol)
    assert first != second


def test_missing_file_hashes_to_none_not_to_the_empty_hash(rm: Any, tmp_path: Path) -> None:
    """An absent file and an empty file are different facts.

    Returning the empty-string SHA would make `present: true, sha256: e3b0c442...`
    indistinguishable from "the file was deleted".
    """
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    missing = tmp_path / "gone.yaml"
    assert rm.sha256_of(empty) == ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    assert rm.sha256_of(missing) is None


def test_dataset_version_counts_both_splits_and_separates_them(rm: Any) -> None:
    """The dataset hash covers train and held-out, and keeps them apart.

    A single pooled hash would let a task move between splits without changing the value,
    which is exactly the isolation Phase 2.3 exists to guarantee.
    """
    dataset = rm.dataset_version()
    assert dataset["present"] is True
    assert set(dataset["splits"]) == {"train", "heldout"}
    assert dataset["splits"]["train"]["sha256"] != dataset["splits"]["heldout"]["sha256"]
    assert dataset["splits"]["train"]["n_tasks"] > 0


# --- honesty about gaps -----------------------------------------------------------------


def test_unresolved_lists_the_known_gaps_and_does_not_hide_them(rm: Any) -> None:
    """The protocol deliberately leaves three values undecided; they must be reported.

    `model.revision` (weights are not pinned to a commit), `sampling.top_p` (never chosen),
    and `grpo.kl_coefficient` (no reference policy exists). If this test starts failing
    because the list is empty, someone filled the gaps - which is good, and the test should
    be updated deliberately rather than deleted.
    """
    manifest = _base(rm)
    assert "protocol.model.revision" in manifest["unresolved"]
    assert "protocol.sampling.top_p" in manifest["unresolved"]
    assert "protocol.grpo.kl_coefficient" in manifest["unresolved"]


def test_no_gpu_is_recorded_as_absent_not_omitted(rm: Any) -> None:
    """Off the training box there is no GPU; the manifest still has to be writable."""
    gpu = _base(rm)["gpu"]
    assert "present" in gpu
    assert isinstance(gpu["present"], bool)
    if not gpu["present"]:
        assert gpu["devices"] == []
        assert gpu["reason"]


def test_a_dirty_tree_is_recorded_as_a_gap(rm: Any, monkeypatch: Any) -> None:
    """A manifest from a dirty tree names a commit that does not contain the code that ran.

    Not an error - refusing here would stop people writing manifests - but it must surface.
    """

    def fake_run(cmd: list[str], cwd: Path = REPO_ROOT, timeout: int = 30) -> str | None:
        if cmd[:2] == ["git", "status"]:
            return " M rewards/compose.py"
        return "deadbeef"

    monkeypatch.setattr(rm, "_run", fake_run)
    manifest = _base(rm)
    assert manifest["git"]["dirty"] is True
    assert any("dirty" in gap for gap in manifest["unresolved"])


def test_outcome_starts_as_planned_with_no_failure_reason(rm: Any) -> None:
    """A manifest is written *before* a run, so it cannot claim an outcome yet."""
    manifest = _base(rm)
    assert manifest["outcome"] == {"status": "planned", "failure_reason": None}
    assert manifest["timing"]["finished_at"] is None


# --- overwrite protection ---------------------------------------------------------------


def test_a_second_identical_manifest_is_not_rewritten(rm: Any, tmp_path: Path) -> None:
    """Re-running the script in a Makefile must be safe."""
    manifest = _base(rm)
    path, wrote = rm.write_manifest(manifest, tmp_path, force=False)
    assert wrote is True
    again, wrote_again = rm.write_manifest(manifest, tmp_path, force=False)
    assert wrote_again is False
    assert path == again


def test_a_differing_manifest_is_refused_without_force(rm: Any, tmp_path: Path) -> None:
    """The core guarantee: an existing result is never silently replaced."""
    first = _base(rm)
    rm.write_manifest(first, tmp_path, force=False)
    changed = _base(rm, seed=7)
    with pytest.raises(FileExistsError, match="--force"):
        rm.write_manifest(changed, tmp_path, force=False)
    # The original is untouched.
    assert json.loads((tmp_path / "manifest.json").read_text())["seed"] == 42


def test_force_preserves_the_superseded_manifest_rather_than_deleting_it(rm: Any, tmp_path: Path) -> None:
    """That the first manifest was wrong is information about the run."""
    first = _base(rm)
    rm.write_manifest(first, tmp_path, force=False)
    changed = _base(rm, seed=7)
    rm.write_manifest(changed, tmp_path, force=True)

    superseded = list(tmp_path.glob("manifest.superseded-*.json"))
    assert len(superseded) == 1
    assert json.loads(superseded[0].read_text())["seed"] == 42
    assert json.loads((tmp_path / "manifest.json").read_text())["seed"] == 7


def test_finish_records_the_end_time_and_failure_reason(rm: Any, tmp_path: Path) -> None:
    """`failure_reason` is required by 5.2; this is where it is written."""
    path, _ = rm.write_manifest(_base(rm), tmp_path, force=False)
    closed = rm.finish_manifest(path, "failed", "CUDA out of memory at step 3")
    assert closed["outcome"]["status"] == "failed"
    assert closed["outcome"]["failure_reason"] == "CUDA out of memory at step 3"
    assert closed["timing"]["finished_at"] is not None
    assert closed["timing"]["duration_s"] is not None


def test_finish_does_not_alter_the_identity_of_the_run(rm: Any, tmp_path: Path) -> None:
    """Closing a run must not become a way to edit what produced it."""
    before = _base(rm)
    path, _ = rm.write_manifest(before, tmp_path, force=False)
    after = rm.finish_manifest(path, "completed", None)
    assert rm._identity(before) == rm._identity(after)


# --- the CLI guard ------------------------------------------------------------------------


def test_cli_refuses_to_manifest_training_on_the_heldout_split(rm: Any, tmp_path: Path) -> None:
    """The seal is enforced at the point a run is declared, not only in the evaluation code.

    `docs/heldout_seal.md`: the held-out split is evaluated exactly once and is never
    trained on. A manifest is where "this run will train on heldout" becomes a fact.
    """
    code = rm.main(
        [
            "--arm",
            "training",
            "--run-id",
            "should_not_exist",
            "--split",
            "heldout",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 2
    assert not (tmp_path / "manifest.json").exists()


def test_cli_writes_a_manifest_and_reports_unresolved_gaps(rm: Any, tmp_path: Path) -> None:
    code = rm.main(["--arm", "baseline", "--run-id", "cli_run", "--out", str(tmp_path)])
    assert code == 0
    written = json.loads((tmp_path / "manifest.json").read_text())
    assert written["run_id"] == "cli_run"
    assert written["arm"] == "baseline"
    assert written["unresolved"]
