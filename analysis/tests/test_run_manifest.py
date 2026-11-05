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


def test_unresolved_reports_the_gaps_that_remain_and_only_those(rm: Any) -> None:
    """The protocol's undecided values are reported; the decided ones are not.

    History: this test used to require all three of `model.revision`, `sampling.top_p` and
    `grpo.kl_coefficient` in `unresolved`. Phase 5.4 resolved the first two - the revision is
    now pinned to a commit sha proven against the local HF cache, and `top_p: 1.0` is now set
    explicitly in both arms instead of inherited from a framework default. So requiring them
    to *stay* listed would now be asserting a gap that has been closed, which is the opposite
    of the test's purpose.

    `grpo.kl_coefficient` legitimately remains: there is no reference policy, so the value
    does not exist and can never be filled in. Its `kl_coefficient_status` records that it is
    a closed question rather than an open one.
    """
    unresolved = _base(rm)["unresolved"]
    assert "protocol.grpo.kl_coefficient" in unresolved
    assert "protocol.model.revision" not in unresolved, "the revision was pinned in Phase 5.4"
    assert "protocol.sampling.top_p" not in unresolved, "top_p was set explicitly in Phase 5.4"


def test_a_resolved_protocol_leaves_no_pilot_blocking_gap(rm: Any) -> None:
    """No remaining gap may be a value a rollout pilot would silently run on a default.

    This is the invariant the four pilot steps were about. `git.dirty` and the absent GPU are
    environmental notes, and the KL null is a not-applicable, not an undecided - none of them
    change which weights or which sampler a run uses. Anything else appearing here means a new
    hole opened in the protocol, and a pilot must not start until it is closed.
    """
    allowed = {
        "protocol.grpo.kl_coefficient",  # no reference policy: absent, not undecided
        "gpu (no GPU detected)",  # true off the training box
        "git.dirty (tree had uncommitted changes)",  # true while editing
    }
    unexpected = set(_base(rm)["unresolved"]) - allowed
    assert not unexpected, f"a protocol value a run depends on is unresolved: {sorted(unexpected)}"


# --- the pilot is an input to the manifest, not a side note -------------------------------


def test_the_pilot_arm_hashes_its_overlay(rm: Any) -> None:
    """A pilot run must name the file that defined it, by hash.

    Hashing only the formal protocol would leave the overlay editable: the manifest would
    still say `protocol_1p5b` while the run executed something else. The overlay hash is
    what makes "this pilot ran these rollouts under these settings" a checkable claim.
    """
    manifest = _base(rm, arm="pilot")
    assert manifest["pilot"]["applies"] is True
    assert manifest["pilot"]["file"]["path"] == rm.PILOT_CONFIG
    assert manifest["pilot"]["file"]["present"] is True
    assert manifest["pilot"]["file"]["sha256"] == rm.sha256_of(rm.REPO_ROOT / rm.PILOT_CONFIG)


def test_the_pilot_arm_records_that_it_updates_nothing(rm: Any) -> None:
    """The pilot's defining property is the absence of an update, so it is recorded."""
    pilot = _base(rm, arm="pilot")["pilot"]
    assert pilot["performs_parameter_update"] is False
    assert pilot["training_iterations"] == 0
    assert pilot["optimizer_enabled"] is False
    assert pilot["expect_output_checkpoint"] is False


def test_the_pilot_arm_records_its_rollout_shape_and_reward_split(rm: Any) -> None:
    """Rollout count, concurrency and the offline-only reward are what the pilot measures."""
    pilot = _base(rm, arm="pilot")["pilot"]
    assert pilot["rollouts_per_task"] is not None
    assert pilot["concurrency"] == 1, "the pilot is serial by design"
    assert pilot["heldout_enabled"] is False
    assert pilot["reward_v2_computed_inline"] is False
    assert pilot["formal_protocol"] == rm.PROTOCOL_CONFIG


def test_only_the_pilot_arm_carries_a_pilot_block(rm: Any) -> None:
    """The block is present on every manifest but marked inapplicable outside the pilot.

    Present-and-false rather than absent, so a reader does not have to infer from a missing
    key whether the overlay was considered and skipped or simply forgotten.
    """
    for arm in ("baseline", "training"):
        block = _base(rm, arm=arm)["pilot"]
        assert block["applies"] is False
        assert block["name"] is None
        assert block["training_iterations"] is None


def test_the_pilot_arm_cannot_open_the_heldout_seal(rm: Any) -> None:
    """A rollout-only run does not produce the frozen checkpoint the seal is evaluated against."""
    assert rm.main(["--arm", "pilot", "--run-id", "x", "--split", "heldout"]) == 2


def test_an_unselected_pilot_task_list_is_reported_as_a_gap(rm: Any) -> None:
    """`tasks: null` is honest, and must surface in `unresolved` rather than read as "all"."""
    gaps = _base(rm, arm="pilot")["unresolved"]
    assert "pilot.tasks (not yet selected)" in gaps
    assert "pilot.tasks (not yet selected)" not in _base(rm, arm="baseline")["unresolved"]


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


# --- the output checkpoint is a promise before a run and a fact after it -------------------


def test_output_checkpoint_is_declared_but_not_hashed_before_the_run(rm: Any, tmp_path: Path) -> None:
    """Before a run the output does not exist, so claiming a hash would be inventing one.

    Hashing here would either fail or, worse, succeed against a leftover file from a
    previous run and record the wrong weights against this run.
    """
    target = tmp_path / "ckpt_never_written.safetensors"
    manifest = _base(rm, output_checkpoint=str(target))
    output = manifest["checkpoints"]["output"]
    assert output["spec"] == str(target)
    assert output["local"] is False
    assert output["sha256"] is None


def test_finish_hashes_the_output_checkpoint_once_it_exists(rm: Any, tmp_path: Path) -> None:
    """After a successful run the output is a fact, and the manifest records it as one."""
    target = tmp_path / "adapter.safetensors"
    path, _ = rm.write_manifest(_base(rm, output_checkpoint=str(target)), tmp_path, force=False)

    # The run happens; the weights appear.
    target.write_bytes(b"not real weights, but a real file")
    closed = rm.finish_manifest(path, "completed", None)

    output = closed["checkpoints"]["output"]
    assert output["local"] is True
    assert output["sha256"] == rm.sha256_of(target)
    assert "reason" not in output, "a present file needs no excuse"


def test_finish_does_not_invent_a_hash_when_the_output_is_absent(rm: Any, tmp_path: Path) -> None:
    """A run reported as completed that wrote no weights must say so, not imply success.

    `sha256: null` with an explicit reason is the honest record. A hash here - of an empty
    string, or carried over from the declaration - would make a missing checkpoint look
    like a reproducible one.
    """
    target = tmp_path / "never_created.safetensors"
    path, _ = rm.write_manifest(_base(rm, output_checkpoint=str(target)), tmp_path, force=False)
    closed = rm.finish_manifest(path, "completed", None)

    output = closed["checkpoints"]["output"]
    assert output["local"] is False
    assert output["sha256"] is None
    assert output["reason"] == "missing_at_close"


def test_an_empty_checkpoint_directory_is_not_reported_as_a_checkpoint(rm: Any, tmp_path: Path) -> None:
    """An existing path is not the same claim as existing weights.

    A checkpoint directory is normal and is now hashed as a tree. But a directory holding
    no files means the run wrote nothing, so it must not read as a successful hash. `None`
    plus `empty_directory` says exactly that: the path exists, the weights do not.
    """
    checkpoint_dir = tmp_path / "adapter_dir"
    checkpoint_dir.mkdir()
    path, _ = rm.write_manifest(_base(rm, output_checkpoint=str(checkpoint_dir)), tmp_path, force=False)
    closed = rm.finish_manifest(path, "completed", None)

    output = closed["checkpoints"]["output"]
    assert output["local"] is True
    assert output["kind"] == "directory"
    assert output["sha256"] is None
    assert output["reason"] == "empty_directory"


def test_a_checkpoint_directory_is_hashed_as_a_tree(rm: Any, tmp_path: Path) -> None:
    """A directory of weights is the normal case, so it must produce a real hash."""
    checkpoint_dir = tmp_path / "adapter_dir"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"weights")
    (checkpoint_dir / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")

    path, _ = rm.write_manifest(_base(rm, output_checkpoint=str(checkpoint_dir)), tmp_path, force=False)
    closed = rm.finish_manifest(path, "completed", None)

    output = closed["checkpoints"]["output"]
    assert output["local"] is True
    assert output["kind"] == "directory"
    assert output["sha256"] == rm.sha256_of_tree(checkpoint_dir)
    assert output["sha256"] is not None
    assert "reason" not in output


def test_a_checkpoint_directory_records_its_shape_not_just_its_hash(rm: Any, tmp_path: Path) -> None:
    """A digest alone cannot distinguish a full save from a truncated one.

    `file_count` and `total_bytes` come from the same traversal as the digest, so a save
    that wrote two of three shards is visible in the record without re-hashing the tree.
    `tree_sha256` repeats the digest under a name that says it covers a directory.
    """
    checkpoint_dir = tmp_path / "adapter_dir"
    checkpoint_dir.mkdir()
    first = b"weights"
    second = b'{"r": 32}'
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(first)
    (checkpoint_dir / "adapter_config.json").write_bytes(second)

    path, _ = rm.write_manifest(_base(rm, output_checkpoint=str(checkpoint_dir)), tmp_path, force=False)
    output = rm.finish_manifest(path, "completed", None)["checkpoints"]["output"]

    assert output["kind"] == "directory"
    assert output["file_count"] == 2
    assert output["total_bytes"] == len(first) + len(second)
    assert output["tree_sha256"] == output["sha256"] == rm.sha256_of_tree(checkpoint_dir)
    assert "reason" not in output


def test_an_empty_checkpoint_directory_records_no_shape(rm: Any, tmp_path: Path) -> None:
    """An empty directory has no file count to report, and must not report zero as if it saved.

    The digest fields are absent rather than hashed-as-empty, so a downstream reader cannot
    mistake `total_bytes: 0` for a legitimate zero-byte checkpoint.
    """
    checkpoint_dir = tmp_path / "empty_dir"
    checkpoint_dir.mkdir()

    path, _ = rm.write_manifest(_base(rm, output_checkpoint=str(checkpoint_dir)), tmp_path, force=False)
    output = rm.finish_manifest(path, "completed", None)["checkpoints"]["output"]

    assert output["kind"] == "directory"
    assert output["reason"] == "empty_directory"
    assert output["sha256"] is None
    for absent in ("tree_sha256", "file_count", "total_bytes"):
        assert absent not in output, f"{absent} must not be recorded for an empty directory"


def test_a_directory_hash_ignores_file_order_and_mtime(rm: Any, tmp_path: Path) -> None:
    """Two byte-identical checkpoints must hash the same even if written differently.

    Filesystem iteration order and timestamps are not properties of the weights. A copy of
    a checkpoint is the same checkpoint, so `cp -r` and a fresh write must agree.
    """
    first = tmp_path / "a"
    second = tmp_path / "b"
    for root, order in ((first, ("z.bin", "a.bin", "m/x.bin")), (second, ("m/x.bin", "a.bin", "z.bin"))):
        for name in order:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(name.encode())

    assert rm.sha256_of_tree(first) == rm.sha256_of_tree(second)


def test_a_directory_hash_changes_when_one_weight_file_changes(rm: Any, tmp_path: Path) -> None:
    """The hash must actually name the contents, or it proves nothing."""
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"weights v1")
    before = rm.sha256_of_tree(checkpoint_dir)

    (checkpoint_dir / "adapter_model.safetensors").write_bytes(b"weights v2")
    assert rm.sha256_of_tree(checkpoint_dir) != before


def test_a_directory_hash_notices_a_renamed_file(rm: Any, tmp_path: Path) -> None:
    """Renaming a shard changes what the checkpoint is, so the path text is part of the hash."""
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "shard_a.bin").write_bytes(b"same bytes")
    before = rm.sha256_of_tree(checkpoint_dir)

    (checkpoint_dir / "shard_a.bin").rename(checkpoint_dir / "shard_b.bin")
    assert rm.sha256_of_tree(checkpoint_dir) != before


def test_an_empty_directory_has_no_tree_hash(rm: Any, tmp_path: Path) -> None:
    """Documented answer for the empty case: no files, no hash."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert rm.sha256_of_tree(empty) is None


def test_finish_on_a_failed_run_reports_absence_without_a_hash(rm: Any, tmp_path: Path) -> None:
    """A crashed run has no output; the manifest must not pretend it does."""
    target = tmp_path / "crashed.safetensors"
    path, _ = rm.write_manifest(_base(rm, output_checkpoint=str(target)), tmp_path, force=False)
    closed = rm.finish_manifest(path, "failed", "CUDA out of memory at step 3")

    assert closed["outcome"]["status"] == "failed"
    assert closed["checkpoints"]["output"]["sha256"] is None
    assert closed["checkpoints"]["output"]["reason"] == "missing_at_close"


def test_finish_tolerates_a_manifest_that_declares_no_output(rm: Any, tmp_path: Path) -> None:
    """The common case - a rollout-only or evaluation run with no checkpoint - must not break."""
    path, _ = rm.write_manifest(_base(rm, output_checkpoint=None), tmp_path, force=False)
    closed = rm.finish_manifest(path, "completed", None)
    assert closed["checkpoints"]["output"] is None


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
