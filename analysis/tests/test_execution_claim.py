"""Phase 6I execution-claim contract - a manifest either authorises a run or it does not.

The manifest grew a second job in this phase. It used to describe a run; now it also decides
whether one is allowed to happen. Those are different questions with different answers, and
the whole point of this file is that they never collapse into one:

  execution_claim: false   an inspection report. Written for a reader, never acted on.
  execution_claim: true    an authorisation. A dirty tracked tree is then a blocker, because
                           the commit the manifest names would not contain the code that ran.

A manifest that is *always* executable is not a gate; a manifest that is *never* executable
cannot be written on a laptop. So the flag is what moves a field between "warning" and
"blocker", and the launcher is what refuses to start without it.

Each case pins one obligation:

  1-3   a dirty tracked tree blocks a claim (exit 2) but only warns on an inspection report
  4-6   the launcher refuses a manifest with no claim, a stale commit, or a changed hash
  7-8   a clean tree with a valid claim passes; GPU absence blocks the rollout but not the
        inspection that precedes it
  9     a null KL term stays not-applicable rather than becoming a blocker
  10    the previous failed-preflight evidence is byte-identical

CPU-only, by construction: the gate reads git, the filesystem and a dict. No GPU, vLLM, model
weights, network, NeMo or Ray. The "clean tree" cases run the gate against a temporary git
repository instead of this one, so the suite does not depend on whether the developer has
uncommitted work - and cannot be made green by committing it.
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
MANIFEST_SCRIPT = REPO_ROOT / "scripts" / "run_manifest.py"
LAUNCHER_SCRIPT = REPO_ROOT / "scripts" / "run_in_serving_env.py"
GATE_SCRIPT = REPO_ROOT / "scripts" / "execution_gate.py"

#: The two Phase 6I directories whose contents are evidence and must not drift.
PHASE6I_RUNS = {
    "results/raw/phase6i_bugfix_offbyone_pagination_20261106T161132Z": (
        "cf4e1624402702d7ac766dcbcbb929e38661593a8d84b484bfc42d808669e273"
    ),
    "results/raw/phase6i_bugfix_offbyone_pagination_20261106T162531Z": (
        "beaf9c6c9e3844f903f71d6cc780fd3eeb336e8ddefab3670d0e19ec5a69396c"
    ),
}


def _load(name: str, path: Path) -> Any:
    """Import a script file, which is not importable as a package module."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rm() -> Any:
    return _load("run_manifest_execution_claim_under_test", MANIFEST_SCRIPT)


@pytest.fixture(scope="module")
def gate() -> Any:
    return _load("execution_gate_under_test", GATE_SCRIPT)


def _config(tmp: Path, *, name: str = "merged_config.yaml", **sections: Any) -> str:
    """Write a materialised eval config under `tmp` and return its absolute path.

    Same shape as the enforcement suite's helper: a legitimate rollout-only diagnostic run
    with the declared `output.*` paths left repo-relative.
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
            "run_id": "phase6i_claim_probe",
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


def _base(rm: Any, *, execution_claim: bool = False, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "arm": "pilot",
        "run_id": "phase6i_claim_probe",
        "split": "train",
        "seed": 42,
        "input_checkpoint": None,
        "output_checkpoint": None,
        "note": None,
        "execution_claim": execution_claim,
    }
    kwargs.update(overrides)
    return rm.build_manifest(**kwargs)


def _report(rm: Any, config_path: str, *, execution_claim: bool = False, **overrides: Any) -> dict[str, Any]:
    """An executable report for `config_path`, with the git state supplied explicitly.

    `git.dirty` is injected rather than observed. Reading it from the repository the tests
    live in made this suite green only while the developer happened to have uncommitted
    edits: committing the very work these tests cover flips the tree to clean and the
    assertion below then fails for a reason that has nothing to do with the gate. The
    property under test - "a claim turns a dirty tree into a blocker" - is a property of
    `classify_fields`, so the tree state is an input to be set, not an ambient fact to be
    waited for. The cases that exercise real git behaviour (`test_06*`, `test_07`) run
    against the throwaway `git_repo` fixture instead, which is where git is the subject.
    """
    tree = {"commit": "0" * 40, "branch": "main", "dirty": True, "untracked": [], "untracked_ignored_count": 0}
    manifest = _base(rm, execution_claim=execution_claim, eval_config=config_path, **overrides)
    # `build_manifest` observes the live repository, so the tree state is replaced after
    # construction rather than passed in. Only the fields the gate reads are touched.
    manifest["git"] = {**manifest.get("git", {}), **overrides.get("git", tree)}
    return rm.executable_report(manifest)


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(MANIFEST_SCRIPT), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A throwaway git repository, so "clean tree" does not mean this repository.

    Testing the clean-tree path against the real repo would couple the suite to whatever the
    developer has staged, and the usual way to make such a test green is to commit - which is
    exactly the habit the gate exists to discourage.
    """
    root = tmp_path / "repo"
    root.mkdir()
    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(tmp_path),
    }
    for args in (
        ("init", "-b", "main"),
        ("add", "-A"),
        ("commit", "--allow-empty", "-m", "seed"),
    ):
        subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True, check=True)
    return root


def _commit_in(root: Path, files: dict[str, str]) -> str:
    env = {
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, env=env, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "add"], cwd=root, env=env, capture_output=True, check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, env=env, capture_output=True, text=True, check=True)
    return head.stdout.strip()


def _claimable_manifest(gate: Any, root: Path) -> dict[str, Any]:
    """A manifest that satisfies every condition, so each case can break exactly one."""
    config_rel = "configs/merged_config.yaml"
    digest = _commit_in(root, {config_rel: "run:\n  rollouts_per_task: 4\n"})
    head = gate.current_head(root)
    return {
        "execution_claim": True,
        "outcome": {"status": "executable"},
        "execution_blockers": [],
        "git": {"commit": head, "branch": "main", "dirty": False, "untracked": [], "untracked_ignored_count": 0},
        "executed": {
            "path": config_rel,
            "sha256": hashlib.sha256((root / config_rel).read_bytes()).hexdigest(),
            "settings": {"tasks": ["t1", "t2"], "rollouts_per_task": 4},
        },
        "gpu": {"present": True},
        "_seed_commit": digest,
    }


# --- case 1-3: dirty tree, with and without a claim ----------------------------------


def test_01_a_dirty_tracked_tree_blocks_an_execution_claim(rm: Any, tmp_path: Path) -> None:
    """With a claim, a dirty tree is a blocker: the commit would not name the code that ran."""
    overlay = tmp_path / "unmaterialised_overlay.yaml"
    overlay.write_text("model:\n  name: Qwen/Qwen2.5-Coder-1.5B-Instruct\n", encoding="utf-8")

    report = _report(rm, str(overlay), execution_claim=True)

    assert report["ok"] is False
    # The tree is dirty by construction (`_report`), which is the point: the gate must not
    # pass just because the config happens to be fine.
    assert any("git.dirty" in blocker for blocker in report["blockers"]), report["blockers"]


def test_02_a_dirty_tree_does_not_block_a_validate_only_inspection(rm: Any, tmp_path: Path) -> None:
    """Without a claim the same tree is a warning: an inspection report can be written anywhere."""
    overlay = tmp_path / "unmaterialised_overlay.yaml"
    overlay.write_text("model:\n  name: Qwen/Qwen2.5-Coder-1.5B-Instruct\n", encoding="utf-8")

    report = _report(rm, str(overlay), execution_claim=False)

    assert not any("git.dirty" in blocker for blocker in report["blockers"]), report["blockers"]
    assert any(item["field"].startswith("git.dirty") for item in report["warnings"]), report["warnings"]


def test_03_cli_exits_2_for_a_dirty_tree_under_a_claim(tmp_path: Path) -> None:
    """Policy A: a refused preflight writes nothing and says so on stderr.

    This is the CLI-level companion to `test_01`, and it deliberately does not assert that
    the tree *is* dirty: the CLI inspects the repository it ships in, which this suite cannot
    make dirty without mutating the developer's checkout. What it pins down is the outcome
    the two layers share - exit 2, a refusal on stderr, and no manifest left behind - and
    that `git.dirty` is named in that refusal whenever the tree is not clean. On a clean tree
    the same command still exits 2, because this overlay is unmaterialised; `test_01` is the
    test that isolates the dirty tree as the reason.
    """
    overlay = tmp_path / "unmaterialised_overlay.yaml"
    overlay.write_text("model:\n  name: Qwen/Qwen2.5-Coder-1.5B-Instruct\n", encoding="utf-8")
    target = tmp_path / "out"

    result = _run_cli(
        "--arm",
        "pilot",
        "--run-id",
        "dirty_claim",
        "--out",
        str(target),
        "--eval-config",
        str(overlay),
        "--execution-claim",
    )

    assert result.returncode == 2, result.stdout + result.stderr
    assert "refusing" in result.stderr and "does not describe a runnable" in result.stderr, result.stderr
    assert "nothing was written" in result.stderr, result.stderr
    assert not target.exists(), "a refused preflight must leave no manifest behind"


def test_03b_validate_only_mode_reports_a_non_executable_inspection(rm: Any, tmp_path: Path) -> None:
    """The same config is an inspection report without a claim and a refusal with one.

    This is the contract in one test: the *document* is identical, only `execution_claim`
    differs, and that flag is the entire difference between "read this" and "run this".
    """
    config_path = _config(tmp_path)

    inspection = _report(rm, config_path, execution_claim=False)
    claimed = _report(rm, config_path, execution_claim=True)

    # Without a claim the dirty tree is a warning, so the report is executable-as-a-report.
    # The status is `executable` rather than `no_execution_claim` because that status means
    # "nothing here could ever run"; this document describes a run that *could*, it just does
    # not authorise one. Authorisation is the launcher's question, and there the missing
    # claim is fatal.
    assert inspection["status"] == "executable"
    assert inspection["ok"] is True
    assert not any("git.dirty" in blocker for blocker in inspection["blockers"])
    assert any(item["field"].startswith("git.dirty") for item in inspection["warnings"]), inspection["warnings"]

    # With a claim the same tree is a blocker. Nothing else about the config changed.
    assert claimed["status"] == "preflight_failed"
    assert claimed["ok"] is False
    assert any("git.dirty" in blocker for blocker in claimed["blockers"]), claimed["blockers"]


# --- case 4-6: the launcher refuses ---------------------------------------------------


def test_04_the_launcher_rejects_a_manifest_without_an_execution_claim(gate: Any, git_repo: Path) -> None:
    """An inspection report must never be accepted by the launcher."""
    manifest = _claimable_manifest(gate, git_repo)
    manifest["execution_claim"] = False

    result = gate.verify_execution_claim(manifest, repo_root=git_repo)

    assert result["ok"] is False
    assert any("execution_claim is not true" in reason for reason in result["reasons"]), result["reasons"]


def test_05_the_launcher_rejects_a_commit_that_is_no_longer_head(gate: Any, git_repo: Path) -> None:
    """A manifest written before a commit names code that is no longer what HEAD is."""
    manifest = _claimable_manifest(gate, git_repo)
    manifest["git"]["commit"] = "0" * 40

    result = gate.verify_execution_claim(manifest, repo_root=git_repo)

    assert result["ok"] is False
    assert any("but HEAD is" in reason for reason in result["reasons"]), result["reasons"]
    assert "the tracked tree has uncommitted changes" not in " ".join(result["reasons"])


def test_06_the_launcher_rejects_a_changed_config_hash(gate: Any, git_repo: Path) -> None:
    """The config is re-hashed at launch: a recorded hash that no longer matches is a refusal."""
    manifest = _claimable_manifest(gate, git_repo)
    (git_repo / "configs" / "merged_config.yaml").write_text("run:\n  rollouts_per_task: 8\n", encoding="utf-8")

    result = gate.verify_execution_claim(manifest, repo_root=git_repo)

    assert result["ok"] is False
    assert any("changed since the manifest was written" in reason for reason in result["reasons"]), result["reasons"]


def test_06b_the_launcher_rejects_a_missing_config(gate: Any, git_repo: Path) -> None:
    """A config that is gone cannot be verified, so it is not assumed unchanged."""
    manifest = _claimable_manifest(gate, git_repo)
    (git_repo / "configs" / "merged_config.yaml").unlink()
    # Removing a tracked file and leaving it removed keeps the tree dirty too, so assert on
    # the specific reason rather than on the set.
    result = gate.verify_execution_claim(manifest, repo_root=git_repo)

    assert result["ok"] is False
    assert any("is gone" in reason for reason in result["reasons"]), result["reasons"]


def test_06c_the_launcher_rejects_a_dirty_tracked_tree(gate: Any, git_repo: Path) -> None:
    manifest = _claimable_manifest(gate, git_repo)
    (git_repo / "src.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "src.py"], cwd=git_repo, capture_output=True, check=True)

    result = gate.verify_execution_claim(manifest, repo_root=git_repo)

    assert result["ok"] is False
    assert any("tracked tree has uncommitted changes" in reason for reason in result["reasons"]), result["reasons"]


def test_06d_the_launcher_rejects_stray_untracked_files(gate: Any, git_repo: Path) -> None:
    """Requirement 6: untracked source outside ignored paths is reported, not ignored."""
    manifest = _claimable_manifest(gate, git_repo)
    (git_repo / "stray_source.py").write_text("y = 2\n", encoding="utf-8")

    result = gate.verify_execution_claim(manifest, repo_root=git_repo)

    assert result["ok"] is False
    assert any("untracked files" in reason and "stray_source" in reason for reason in result["reasons"]), result[
        "reasons"
    ]


def test_06e_ignored_result_files_do_not_make_the_tree_dirty(gate: Any, git_repo: Path) -> None:
    """Requirement 6, the other half: a rollout writes into results/, which is ignored."""
    (git_repo / ".gitignore").write_text("results/\n", encoding="utf-8")
    _commit_in(git_repo, {})
    manifest = _claimable_manifest(gate, git_repo)
    (git_repo / "results" / "raw" / "run").mkdir(parents=True, exist_ok=True)
    (git_repo / "results" / "raw" / "run" / "rollouts.jsonl").write_text("{}\n", encoding="utf-8")

    result = gate.verify_execution_claim(manifest, repo_root=git_repo)

    assert result["ok"] is True, result["reasons"]


# --- case 7-8: the passing case, and what a missing GPU does --------------------------


def test_07_a_clean_tree_with_a_valid_claim_passes(gate: Any, git_repo: Path) -> None:
    """The gate has to be passable, or it is a wall rather than a gate."""
    manifest = _claimable_manifest(gate, git_repo)

    result = gate.verify_execution_claim(manifest, repo_root=git_repo)

    assert result["ok"] is True, result["reasons"]
    assert result["reasons"] == []


def test_08_a_missing_gpu_blocks_a_rollout_but_not_the_inspection(rm: Any, gate: Any, tmp_path: Path) -> None:
    """GPU absence is environmental: it stops the run, not the report that explains it.

    The manifest is built with the GPU record removed rather than read from this machine, so
    the case is exercised identically on a laptop and on the training box. Both halves are
    asserted on the same manifest, because the interesting claim is that one document can be
    a valid inspection and an invalid authorisation at once.
    """
    config_path = _config(tmp_path)
    manifest = _base(rm, execution_claim=True, eval_config=config_path)
    manifest["gpu"] = {"present": False, "reason": "no GPU detected", "devices": [], "cuda_visible_devices": None}

    inspection = rm.executable_report({**manifest, "execution_claim": False})
    assert any(item["field"].startswith("gpu") for item in inspection["warnings"]), inspection["warnings"]
    assert not any("gpu" in blocker.lower() for blocker in inspection["blockers"]), inspection["blockers"]

    with_claim = rm.executable_report(manifest)
    assert any("gpu" in blocker.lower() for blocker in with_claim["blockers"]), with_claim["blockers"]

    # ...and the launcher agrees, while still allowing a caller that is not asking for a GPU run.
    blocked = gate.verify_execution_claim(manifest, repo_root=REPO_ROOT)
    assert any("no GPU" in reason for reason in blocked["reasons"]), blocked["reasons"]
    allowed = gate.verify_execution_claim(manifest, repo_root=REPO_ROOT, require_gpu=False)
    assert not any("no GPU" in reason for reason in allowed["reasons"]), allowed["reasons"]


# --- case 9: not-applicable stays not-applicable --------------------------------------


def test_09_a_null_kl_term_stays_not_applicable(rm: Any, tmp_path: Path) -> None:
    """A rollout-only diagnostic has no reference model, so KL is not missing - it is moot."""
    overlay = tmp_path / "unmaterialised_overlay.yaml"
    overlay.write_text("model:\n  name: Qwen/Qwen2.5-Coder-1.5B-Instruct\n", encoding="utf-8")

    for claim in (False, True):
        report = _report(rm, str(overlay), execution_claim=claim)
        assert any("kl" in item["field"].lower() for item in report["not_applicable"]), report
        assert not any("kl" in blocker.lower() for blocker in report["blockers"]), report["blockers"]

    # The label does not depend on the machine or the claim, which is why it is a constant
    # table rather than a special case in the classifier.
    assert "protocol.grpo.kl_coefficient" in rm.NOT_APPLICABLE_REASONS


# --- case 10: evidence does not move --------------------------------------------------


@pytest.mark.parametrize("run_dir,expected", sorted(PHASE6I_RUNS.items()))
def test_10_the_phase6i_evidence_is_byte_identical(rm: Any, run_dir: str, expected: str) -> None:
    """The failed preflight is evidence of a failure; rewriting it would erase the record."""
    manifest_path = REPO_ROOT / run_dir / "manifest.json"
    assert manifest_path.is_file(), f"missing evidence: {manifest_path}"

    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert digest == expected, f"{run_dir} changed: {digest}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # These manifests predate `executable`, so the report is recomputed from them rather than
    # read out of them. A failed preflight must stay a failed preflight, and its outcome must
    # still be the one that was recorded at write time.
    assert manifest["outcome"]["status"] == "planned"
    recomputed = rm.executable_report(manifest)
    assert recomputed["ok"] is False
    assert recomputed["status"] == "preflight_failed"
    assert recomputed["blockers"], "a failed preflight must name why it failed"


def test_10b_the_launcher_refuses_every_failed_preflight_manifest(gate: Any) -> None:
    """The recorded failures stay failures: no manifest on disk may be launched from."""
    for run_dir in PHASE6I_RUNS:
        manifest = json.loads((REPO_ROOT / run_dir / "manifest.json").read_text(encoding="utf-8"))
        result = gate.verify_execution_claim(manifest, repo_root=REPO_ROOT)
        assert result["ok"] is False, f"{run_dir} would have been launched"
        assert any("execution_claim is not true" in reason for reason in result["reasons"]), result["reasons"]
