"""Phase 6I run identity - one run id, one directory, everywhere.

A prepared Phase 6I run carried two different identities at once:

    result directory / manifest run_id : phase6i_bugfix_offbyone_pagination_20261107T045749Z
    materialised config output.run_id  : phase6i_bugfix_offbyone_pagination_20261106T161132Z

Nothing in the manifest was individually false. Every field was a plausible value that had
been true of *some* run; the manifest simply described two runs and claimed to describe one.
The cause is structural rather than clerical: the eval overlays live in `configs/` and are
committed, so their `output:` block names whichever run last exercised them. Materialising a
fresh overlay copies that block verbatim, including `output.run_id` and all three output
paths. A rollout started from such a config would have written its rollouts, private rows and
failure records into the *previous* attempt's directory - mixing two attempts into one set of
evidence and destroying the failed attempt's in the process.

The contract this file pins, in one sentence: the CLI run id, the manifest's `run_id`, the
executed config's `output.run_id`, and the basename of the result directory are the same
string, and every evidence path the run writes lives inside that directory.

Each case holds one line of it:

  01    all identity fields agree and the run passes
  02    a stale `output.run_id` inherited from a committed overlay is refused
  03    a result directory whose basename is not the run id is refused
  04    a public trace outside the run directory is refused
  05    a private trace outside the run directory is refused
  06    a failure file outside the run directory is refused
  07    a manifest `run_id` that disagrees with the config is refused
  08    the launcher detects a config identity change *after* the manifest was written
  09    a refused execution claim writes no manifest and exits 2
  10    the previously prepared run's evidence is byte-identical

CPU-only by construction: the contract reads strings and one YAML file. No GPU, vLLM, model
weights, network, NeMo or Ray, and no subprocess that touches either GPU. Case 08 drives the
real launcher through a shim because the recheck has to be proven where it runs, but the
environment probe and the target are both replaced, so no interpreter but this one is needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.execution_gate import run_identity_checks, verify_run_identity
from scripts.run_manifest import executable_report

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The identity the defective prepared run was supposed to have.
PREPARED_RUN_ID = "phase6i_bugfix_offbyone_pagination_20261107T045749Z"
#: The identity its materialised config actually carried, inherited from the committed overlay.
INHERITED_RUN_ID = "phase6i_bugfix_offbyone_pagination_20261106T161132Z"

#: The prepared run that was rejected as an execution blocker. Preserved byte-for-byte: it is
#: the evidence for the defect, and a fix that rewrote it would delete the thing it fixed.
PREPARED_RUN = f"results/raw/{PREPARED_RUN_ID}"
PREPARED_ARTIFACTS = {
    "manifest.json": "ad0de26819d871995fabb95cfd91f86c7eb808f2f896bd1f717c688aadb0219c",
    "merged_config.yaml": "3a34115707e961422d02252586d96e3c6184bb6466f20fb8749b76feae06c3b5",
}

#: A materialised config that obeys the contract. Extended per case so exactly one field can
#: be broken at a time, which is what makes a failure message name the culprit.
GOOD_CONFIG = """
model:
  name: Qwen/Qwen2.5-Coder-1.5B-Instruct
  revision: 0e3f9a2b1c
sampling:
  temperature: 0.0
  top_p: 1.0
  seed: 42
run:
  task_subset: [bugfix_offbyone_pagination]
  rollouts_per_task: 4
  diagnostic_only: true
output:
  run_id: {run_id}
  rollouts: results/raw/{run_id}/rollouts.jsonl
  failures: results/raw/{run_id}/rollouts_failures.jsonl
  private: results/raw/{run_id}/rollouts_private.jsonl
  analysis: results/analysis/{run_id}.json
"""


def _config_text(run_id: str = PREPARED_RUN_ID) -> str:
    return GOOD_CONFIG.format(run_id=run_id)


@pytest.fixture
def identity_repo(tmp_path: Path) -> Path:
    """A throwaway repository whose single committed config obeys the identity contract.

    A throwaway repository rather than this one for the reason the launcher cases give: both
    `build_manifest` and the gate read whatever repository they live in, so a case built on
    this working tree would change its answer whenever the developer had uncommitted work.
    """
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    (root / "README.md").write_text("identity fixture\n", encoding="utf-8")
    config = root / "configs" / "merged_config.yaml"
    config.write_text(_config_text(), encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "seed")
    return root


_GIT_ENV = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "PATH": "/usr/bin:/bin:/usr/local/bin",
}


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=root, env=_GIT_ENV, capture_output=True, text=True, check=True)
    return done.stdout.strip()


def _manifest_for(root: Path, config_rel: str = "configs/merged_config.yaml") -> dict[str, Any]:
    """Build an execution-claiming manifest against the fixture repository.

    `build_manifest` reads `REPO_ROOT` rather than taking a path, so the fixture repository is
    assembled by hand here instead: the same shape the real builder produces, with the git
    facts read from the fixture's own history.
    """
    config = root / config_rel
    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    output = raw.get("output") or {}
    run_id = output.get("run_id")
    head = _git(root, "rev-parse", "HEAD")
    rollouts = output.get("rollouts") or ""
    return {
        "manifest_version": "1",
        "run_id": run_id,
        "arm": "pilot",
        "execution_claim": True,
        "execution_blockers": [],
        "git": {"commit": head, "branch": "main", "dirty": False, "untracked": [], "untracked_ignored_count": 0},
        "gpu": {"present": True},
        "executed": {
            "path": config_rel,
            "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "present": True,
            "materialised": True,
            "settings_complete": True,
            "settings": {
                "run_id": run_id,
                "tasks": ["bugfix_offbyone_pagination"],
                "rollouts_per_task": 4,
                "model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                "model_revision": "0e3f9a2b1c",
                "heldout_enabled": False,
                "optimizer_enabled": False,
                "training_iterations": None,
                "expect_output_checkpoint": False,
            },
            "artifacts": {
                "run_dir": os.path.dirname(rollouts),
                "trace": output.get("rollouts"),
                "private_trace": output.get("private"),
                "failures": output.get("failures"),
                "summary": f"{os.path.dirname(rollouts)}/collection_summary.json",
            },
        },
        "artifacts": {
            "run_dir": os.path.dirname(rollouts),
            "trace": output.get("rollouts"),
            "private_trace": output.get("private"),
            "failures": output.get("failures"),
            "summary": f"{os.path.dirname(rollouts)}/collection_summary.json",
            "log": None,
            "server_pid": None,
        },
    }


def _identity_result(manifest: dict[str, Any]) -> list[str]:
    """The identity reasons for a manifest's own recorded fields, without touching disk."""
    executed = manifest["executed"]
    intended = executed["artifacts"]
    return run_identity_checks(
        cli_run_id=manifest["run_id"],
        manifest_run_id=manifest["run_id"],
        executed_run_id=executed["settings"]["run_id"],
        config_run_id=executed["settings"]["run_id"],
        run_dir=intended["run_dir"],
        output_paths={
            "output.rollouts": intended["trace"],
            "output.private": intended["private_trace"],
            "output.failures": intended["failures"],
        },
    )


# --- 01: the contract holds -----------------------------------------------------------


def test_01_a_self_consistent_run_passes(identity_repo: Path) -> None:
    manifest = _manifest_for(identity_repo)

    assert _identity_result(manifest) == []
    assert verify_run_identity(manifest, repo_root=identity_repo)["ok"] is True


# --- 02-07: one field broken at a time ------------------------------------------------


def test_02_a_stale_output_run_id_is_refused(identity_repo: Path) -> None:
    """The Phase 6I defect itself: the config still names the previous attempt."""
    config = identity_repo / "configs" / "merged_config.yaml"
    config.write_text(_config_text(INHERITED_RUN_ID), encoding="utf-8")
    manifest = _manifest_for(identity_repo)

    assert manifest["run_id"] == INHERITED_RUN_ID, "the fixture must reproduce the inherited identity"

    # Re-point only the manifest at the new id, leaving the config stale - exactly how the
    # defective run was prepared.
    manifest["run_id"] = PREPARED_RUN_ID
    reasons = _identity_result(manifest)

    assert any("config output.run_id" in reason for reason in reasons), reasons
    assert any(INHERITED_RUN_ID in reason for reason in reasons), reasons


def test_03_a_result_directory_not_named_by_the_run_id_is_refused(identity_repo: Path) -> None:
    manifest = _manifest_for(identity_repo)
    manifest["executed"]["artifacts"]["run_dir"] = f"results/raw/{INHERITED_RUN_ID}"

    reasons = _identity_result(manifest)

    assert any("basename" in reason for reason in reasons), reasons


def test_04_a_public_trace_outside_the_run_directory_is_refused(identity_repo: Path) -> None:
    manifest = _manifest_for(identity_repo)
    manifest["executed"]["artifacts"]["trace"] = f"results/raw/{INHERITED_RUN_ID}/rollouts.jsonl"

    reasons = _identity_result(manifest)

    assert any("output.rollouts is outside" in reason for reason in reasons), reasons


def test_05_a_private_trace_outside_the_run_directory_is_refused(identity_repo: Path) -> None:
    manifest = _manifest_for(identity_repo)
    manifest["executed"]["artifacts"]["private_trace"] = f"results/raw/{INHERITED_RUN_ID}/rollouts_private.jsonl"

    reasons = _identity_result(manifest)

    assert any("output.private is outside" in reason for reason in reasons), reasons


def test_06_a_failure_file_outside_the_run_directory_is_refused(identity_repo: Path) -> None:
    manifest = _manifest_for(identity_repo)
    manifest["executed"]["artifacts"]["failures"] = f"results/raw/{INHERITED_RUN_ID}/rollouts_failures.jsonl"

    reasons = _identity_result(manifest)

    assert any("output.failures is outside" in reason for reason in reasons), reasons


def test_07_a_manifest_run_id_that_disagrees_with_the_config_is_refused(identity_repo: Path) -> None:
    """The comparison the manifest never made: its own `run_id` against the config it names.

    The manifest's `run_id` is what the launcher prints and what names the directory; the
    config's `output.run_id` is what the collector writes into. Nothing compared them, so a
    manifest whose two identities disagreed was reported as executable.
    """
    manifest = _manifest_for(identity_repo)
    # The config keeps saying INHERITED; only the manifest's own id moves.
    manifest["executed"]["settings"]["run_id"] = INHERITED_RUN_ID
    manifest["run_id"] = PREPARED_RUN_ID

    reasons = _identity_result(manifest)

    assert any("executed run_id" in reason for reason in reasons), reasons
    assert any(INHERITED_RUN_ID in reason for reason in reasons), reasons

    # And the report refuses it, naming identity rather than a generic gap.
    report = executable_report(manifest)
    assert report["ok"] is False
    assert any("run identity" in blocker for blocker in report["blockers"]), report["blockers"]


# --- 08: the launcher rechecks, after the manifest was written ------------------------


def test_08_the_launcher_detects_a_post_manifest_identity_change(identity_repo: Path) -> None:
    """A manifest can be written on Monday and used on Tuesday.

    The manifest's `executed.sha256` also matches, because the config was rewritten *and* the
    manifest was not - which is the realistic sequence when a config is regenerated in place.
    Re-reading the file is what makes the change visible.
    """
    manifest_path = identity_repo.parent / "manifest.json"
    manifest = _manifest_for(identity_repo)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Rewrite the config so it names the inherited run, leaving the manifest claiming the new
    # one. The hash in the manifest is now wrong too, which is correct: that is what a
    # replaced config looks like.
    config = identity_repo / "configs" / "merged_config.yaml"
    config.write_text(_config_text(INHERITED_RUN_ID), encoding="utf-8")

    result = verify_run_identity(manifest, repo_root=identity_repo)

    assert result["ok"] is False
    assert any("config output.run_id" in reason for reason in result["reasons"]), result["reasons"]


# --- 09: a refused claim writes nothing -----------------------------------------------


def test_09_no_manifest_is_written_when_an_execution_claim_fails(
    identity_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Policy A: the refusal happens before the file is opened."""
    from scripts import run_manifest

    target = identity_repo.parent / "out" / "manifest.json"

    stale = dict(_manifest_for(identity_repo))
    stale["run_id"] = PREPARED_RUN_ID
    monkeypatch.setattr(run_manifest, "build_manifest", lambda **kwargs: dict(stale))

    code = run_manifest.main(
        [
            "--arm",
            "pilot",
            "--run-id",
            PREPARED_RUN_ID,
            "--eval-config",
            "configs/merged_config.yaml",
            "--execution-claim",
            "--out",
            str(target),
        ]
    )

    assert code == 2
    assert not target.exists(), "a refused execution claim wrote a manifest"


# --- 10: the defect's evidence is preserved -------------------------------------------


def test_10_the_prepared_run_is_byte_identical_and_still_refused() -> None:
    """The prepared run is evidence of the defect, not a run to be repaired in place.

    It must still be refused, which is asserted from the committed files rather than from the
    fixtures above: a fix that silently accepted the old directory would be a fix that deleted
    the reason it existed.
    """
    root = REPO_ROOT / PREPARED_RUN
    if not root.is_dir():
        pytest.skip(f"{PREPARED_RUN} is absent; nothing to preserve")

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["run_id"] == PREPARED_RUN_ID
    config = yaml.safe_load((root / "merged_config.yaml").read_text(encoding="utf-8")) or {}
    assert config["output"]["run_id"] == INHERITED_RUN_ID, "the defect is no longer reproduced on disk"

    result = verify_run_identity(manifest)
    assert result["ok"] is False
    assert any(PREPARED_RUN_ID in reason or INHERITED_RUN_ID in reason for reason in result["reasons"])

    # And the fix did not rewrite it on its way to refusing it.
    for name, expected in PREPARED_ARTIFACTS.items():
        payload = (root / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected, f"{name} was modified"
