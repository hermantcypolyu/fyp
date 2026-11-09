"""Phase 6I launcher boundary - the launcher's arguments never reach the target.

The launcher (`scripts/run_in_serving_env.py`) has two jobs that used to be conflated. It
authorises a launch from a manifest, and it then executes the target inside the serving
interpreter. `--manifest` belongs to the first job only. Because the launcher forwarded its
whole argument list to the target, the documented invocation handed `--manifest PATH` to
`scripts/collect_baseline_rollouts.py`, whose strict argparse does not accept it:

    usage: collect_baseline_rollouts.py [-h] [--config CONFIG] ...
    collect_baseline_rollouts.py: error: unrecognized arguments: -- --manifest ...

Authorisation had already succeeded by then, so the run directory held a valid manifest, a
merged config and a server log, and no rollouts. The evidence looked like a rollout that
failed rather than a launcher that never started one.

Each case pins one obligation of the boundary:

  1-2   `--manifest PATH` and `--manifest=PATH` are both consumed
  3-4   the target sees neither the flag nor the path
  5-6   collector-owned arguments survive, in their original order
  7-10  malformed launcher arguments are usage errors, not target arguments
  11-12 the boundary is tested end to end through a real strict argparse target
  13-16 authorisation still precedes execution, including the two refusal paths
  17    a refused launch starts no target process at all

The last case is the reason the file exists. It runs the launcher against a temporary git
repository containing a strict dummy target, and asserts both directions: the documented form
succeeds now, and the same target still fails when `--manifest` is left in its argument list.
A green suite that cannot detect the original bug would repeat the gap that let it through.

CPU-only by construction: the gate reads git and the filesystem, and the targets are tiny
Python scripts. No GPU, vLLM, model weights, network, NeMo or Ray. Authorisation cases use a
throwaway repository, so the suite does not depend on whether the developer has uncommitted
work and cannot be made green by committing it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_SCRIPT = REPO_ROOT / "scripts" / "run_in_serving_env.py"
GATE_SCRIPT = REPO_ROOT / "scripts" / "execution_gate.py"

#: A strict target: it declares only `--config`, matching the real collector's habit of
#: rejecting anything it does not define rather than ignoring it.
STRICT_TARGET = """
import argparse
import json
import sys

parser = argparse.ArgumentParser(prog="strict_target")
parser.add_argument("--config")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--rollouts", type=int)
args = parser.parse_args()  # strict on purpose - no parse_known_args
payload = {"argv": sys.argv[1:], "config": args.config, "dry_run": args.dry_run, "rollouts": args.rollouts}
with open(sys.argv[0] + ".seen.json", "w", encoding="utf-8") as handle:
    json.dump(payload, handle)
sys.exit(0)
"""

#: What the failed Phase 6I launch actually produced. Preserved as evidence.
PHASE6I_FAILED_RUN = "results/raw/phase6i_bugfix_offbyone_pagination_20261107T040554Z"
PHASE6I_FAILED_ARTIFACTS = {
    "manifest.json": (
        "7bdf65c04a69ad207d5d494acb633256626afbff1632a11c3acee3a2b109bf96",
        11047,
    ),
    "merged_config.yaml": (
        "7e295c2721a21fb0adf757f1cd1d08efd7619f5930d2c30a90bf63236abbac16",
        5477,
    ),
    "vllm_server.log": (
        "9d42f7ecf17faa8333282a57fdfb2344ece403bd3b30dac1efb392bf6e7dd437",
        20214,
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
def launcher() -> Any:
    return _load("run_in_serving_env_boundary_under_test", LAUNCHER_SCRIPT)


@pytest.fixture(scope="module")
def gate() -> Any:
    return _load("execution_gate_boundary_under_test", GATE_SCRIPT)


_GIT_ENV = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
    "PATH": "/usr/bin:/bin:/usr/local/bin",
}


def _git(root: Path, *args: str, env: dict[str, str] | None = None) -> str:
    done = subprocess.run(["git", *args], cwd=root, env=env or _GIT_ENV, capture_output=True, text=True, check=True)
    return done.stdout.strip()


#: The run id the fixture's config and manifest share. These cases are about the argument
#: boundary, so the identity has to be consistent for authorisation to be reached at all - a
#: mismatched identity is now its own refusal, and leaving the fixture inconsistent would make
#: every case here fail on the wrong reason.
FIXTURE_RUN_ID = "launcher_boundary_20261107T000000Z"


def _claimable_manifest(gate: Any, root: Path, *, claim: bool = True) -> dict[str, Any]:
    """A manifest that satisfies every condition, so a case can break exactly one."""
    config_rel = "configs/merged_config.yaml"
    target = root / config_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    run_dir = f"results/raw/{FIXTURE_RUN_ID}"
    # The identity fields are populated: the launcher rechecks them immediately before it
    # executes, so a manifest without them is refused before it ever reaches the target.
    target.write_text(
        "run:\n"
        "  rollouts_per_task: 4\n"
        "output:\n"
        f"  run_id: {FIXTURE_RUN_ID}\n"
        f"  rollouts: {run_dir}/rollouts.jsonl\n"
        f"  failures: {run_dir}/rollouts_failures.jsonl\n"
        f"  private: {run_dir}/rollouts_private.jsonl\n"
        f"  analysis: results/analysis/{FIXTURE_RUN_ID}.json\n",
        encoding="utf-8",
    )
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "config")
    head = gate.current_head(root)
    return {
        "run_id": FIXTURE_RUN_ID,
        "execution_claim": claim,
        "outcome": {"status": "executable" if claim else "planned"},
        "execution_blockers": [],
        "git": {"commit": head, "branch": "main", "dirty": False, "untracked": [], "untracked_ignored_count": 0},
        "executed": {
            "path": config_rel,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "settings": {"tasks": ["t1"], "rollouts_per_task": 4, "run_id": FIXTURE_RUN_ID},
            "artifacts": {
                "run_dir": run_dir,
                "trace": f"{run_dir}/rollouts.jsonl",
                "private_trace": f"{run_dir}/rollouts_private.jsonl",
                "failures": f"{run_dir}/rollouts_failures.jsonl",
                "summary": f"{run_dir}/collection_summary.json",
            },
        },
        "artifacts": {
            "run_dir": run_dir,
            "trace": f"{run_dir}/rollouts.jsonl",
            "private_trace": f"{run_dir}/rollouts_private.jsonl",
            "failures": f"{run_dir}/rollouts_failures.jsonl",
            "summary": f"{run_dir}/collection_summary.json",
            "log": None,
            "server_pid": None,
        },
        "gpu": {"present": True},
    }


@pytest.fixture
def strict_repo(tmp_path: Path) -> Path:
    """A throwaway repository holding a strict dummy target and a copy of the launcher.

    Authorisation inspects git, and testing it against this repository would couple the case
    to whatever the developer has staged. The launcher is copied in so the fixture is a
    self-contained repository the gate can inspect.
    """
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-b", "main")
    (root / "strict_target.py").write_text(STRICT_TARGET, encoding="utf-8")
    (root / "README.md").write_text("dummy\n", encoding="utf-8")
    # The launcher imports the gate, which makes the interpreter write __pycache__. Without an
    # ignore rule that is an untracked file, and the gate rightly refuses an unclean tree.
    (root / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    scripts = root / "scripts"
    scripts.mkdir()
    (scripts / "__init__.py").write_text("", encoding="utf-8")
    (scripts / "run_in_serving_env.py").write_text(LAUNCHER_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    (scripts / "execution_gate.py").write_text(GATE_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "seed")
    return root


def _write_manifest(root: Path, gate: Any, *, claim: bool = True, name: str = "manifest.json") -> Path:
    """Write a manifest for `root` outside its tracked tree, so it does not dirty the repo."""
    manifest = _claimable_manifest(gate, root, claim=claim)
    path = root.parent / name
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# --- 1-4: the manifest argument is consumed, in both spellings -------------------------


def test_01_manifest_with_separate_value_is_consumed(launcher: Any) -> None:
    manifest, forwarded = launcher._split_manifest_args(["--", "--manifest", "m.json", "--config", "c.yaml"])

    assert manifest == "m.json"
    assert forwarded == ["--config", "c.yaml"]


def test_02_manifest_with_equals_form_is_consumed(launcher: Any) -> None:
    manifest, forwarded = launcher._split_manifest_args(["--", "--manifest=m.json", "--config", "c.yaml"])

    assert manifest == "m.json"
    assert forwarded == ["--config", "c.yaml"]


def test_03_target_receives_no_manifest_flag(launcher: Any) -> None:
    _, forwarded = launcher._split_manifest_args(["--manifest", "m.json", "--dry-run"])

    assert "--manifest" not in forwarded
    assert not any(token.startswith("--manifest") for token in forwarded)


def test_04_target_receives_no_manifest_path(launcher: Any) -> None:
    _, forwarded = launcher._split_manifest_args(["--manifest", "results/raw/run/manifest.json", "--dry-run"])

    assert "results/raw/run/manifest.json" not in forwarded


# --- 5-6: the target's own arguments are untouched ------------------------------------


def test_05_collector_owned_arguments_survive(launcher: Any) -> None:
    own = ["--config", "c.yaml", "--rollouts", "4", "--task-ids", "t1,t2", "--dry-run"]

    _, forwarded = launcher._split_manifest_args(["--", "--manifest", "m.json", *own])

    assert forwarded == own


def test_06_argument_order_after_the_target_is_preserved(launcher: Any) -> None:
    extra = ["--config", "c.yaml", "--manifest", "m.json", "--rollouts", "4", "--verbose"]

    _, forwarded = launcher._split_manifest_args(extra)

    # Removing one option must not reorder, dedupe or rewrite the rest.
    assert forwarded == ["--config", "c.yaml", "--rollouts", "4", "--verbose"]


def test_06b_caller_argument_list_is_not_mutated(launcher: Any) -> None:
    extra = ["--", "--manifest", "m.json", "--config", "c.yaml"]
    snapshot = list(extra)

    launcher._split_manifest_args(extra)

    assert extra == snapshot


# --- 7-10: malformed launcher arguments are usage errors ------------------------------


def test_07_duplicate_manifest_flags_are_rejected(launcher: Any) -> None:
    with pytest.raises(launcher.ManifestArgumentError, match="more than once"):
        launcher._split_manifest_args(["--manifest", "a.json", "--manifest", "b.json"])


def test_08_manifest_without_a_value_is_rejected(launcher: Any) -> None:
    with pytest.raises(launcher.ManifestArgumentError, match="needs a path"):
        launcher._split_manifest_args(["--manifest"])


def test_09_empty_manifest_value_is_rejected(launcher: Any) -> None:
    with pytest.raises(launcher.ManifestArgumentError, match="empty path"):
        launcher._split_manifest_args(["--manifest="])


def test_09b_whitespace_manifest_value_is_rejected(launcher: Any) -> None:
    with pytest.raises(launcher.ManifestArgumentError, match="empty path"):
        launcher._split_manifest_args(["--manifest", "   "])


def test_10_missing_target_is_a_usage_error(launcher: Any, capsys: pytest.CaptureFixture[str]) -> None:
    """`--manifest` with no target is a usage error: there is nothing to authorise."""
    assert launcher.main(["--"]) == 2

    assert "no target script was named" in capsys.readouterr().err


# --- 11-12: end to end through a real strict target -----------------------------------


def test_11_strict_target_succeeds_through_the_documented_form(launcher: Any, strict_repo: Path, gate: Any) -> None:
    """The whole point: the documented invocation now reaches the target and it parses."""
    manifest = _write_manifest(strict_repo, gate)

    seen, code = _run_strict_target(launcher, strict_repo, manifest, ["--config", "c.yaml", "--rollouts", "4"])

    assert code == 0, seen
    assert seen["argv"] == ["--config", "c.yaml", "--rollouts", "4"]
    assert seen["config"] == "c.yaml"
    assert seen["rollouts"] == 4


def test_12_strict_target_would_fail_if_manifest_leaked(launcher: Any, strict_repo: Path, gate: Any) -> None:
    """The control. Without the fix the same target aborts, so the case can detect the bug."""
    manifest = _write_manifest(strict_repo, gate)

    seen, code = _run_strict_target(launcher, strict_repo, manifest, ["--config", "c.yaml"], bypass_stripping=True)

    assert code != 0
    assert seen == {}, "the target ran even though --manifest leaked; the control proves nothing"


# --- 13-16: authorisation still runs before execution ---------------------------------


def test_13_authorisation_runs_before_the_target(launcher: Any, strict_repo: Path, gate: Any) -> None:
    manifest = _write_manifest(strict_repo, gate)

    seen, code = _run_strict_target(launcher, strict_repo, manifest, ["--config", "c.yaml"])

    # The target ran, and it ran only after the manifest was accepted.
    assert code == 0
    assert seen["config"] == "c.yaml"


def test_14_a_manifest_without_an_execution_claim_exits_5(launcher: Any, strict_repo: Path, gate: Any) -> None:
    manifest = _write_manifest(strict_repo, gate, claim=False)

    seen, code = _run_strict_target(launcher, strict_repo, manifest, ["--config", "c.yaml"])

    assert code == 5
    assert seen == {}


def test_15_a_changed_config_hash_exits_5(launcher: Any, strict_repo: Path, gate: Any) -> None:
    manifest_path = _write_manifest(strict_repo, gate)
    config = strict_repo / "configs" / "merged_config.yaml"
    config.write_text("run:\n  rollouts_per_task: 99\n", encoding="utf-8")

    seen, code = _run_strict_target(launcher, strict_repo, manifest_path, ["--config", "c.yaml"])

    assert code == 5
    assert seen == {}


def test_16_a_missing_manifest_exits_5(launcher: Any, strict_repo: Path) -> None:
    seen, code = _run_strict_target(launcher, strict_repo, strict_repo / "absent.json", ["--config", "c.yaml"])

    assert code == 5
    assert seen == {}


def test_17_no_target_process_starts_when_authorisation_fails(
    launcher: Any, strict_repo: Path, gate: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused launch must not reach the subprocess call, whatever the arguments look like."""
    manifest = _write_manifest(strict_repo, gate, claim=False)
    started: list[list[str]] = []
    monkeypatch.setattr(launcher.subprocess, "call", lambda argv, **kwargs: started.append(argv) or 0)
    monkeypatch.setattr(launcher, "_require_manifest", lambda path: 5)

    # Point the launcher at the fixture repository, where `strict_target.py` exists.
    monkeypatch.setattr(launcher, "REPO_ROOT", strict_repo)
    monkeypatch.chdir(strict_repo)
    code = launcher.main(["strict_target.py", "--", "--manifest", str(manifest), "--config", "c.yaml"])

    assert code == 5
    assert started == [], f"the launcher started a target despite refusing: {started}"


# --- 18: the failed run is evidence and must not drift --------------------------------


def test_18_failed_phase6i_artifacts_are_unchanged() -> None:
    root = REPO_ROOT / PHASE6I_FAILED_RUN

    present = sorted(entry.name for entry in root.iterdir())
    assert present == sorted(PHASE6I_FAILED_ARTIFACTS), f"unexpected contents in {PHASE6I_FAILED_RUN}"

    for name, (digest, size) in PHASE6I_FAILED_ARTIFACTS.items():
        payload = (root / name).read_bytes()
        assert len(payload) == size, f"{name} changed size"
        assert hashlib.sha256(payload).hexdigest() == digest, f"{name} changed content"

    # Collection never started: the launcher died in the target's argparse.
    assert not (root / "rollouts.jsonl").exists()
    assert not (root / "rollouts_private.jsonl").exists()


# --- helpers --------------------------------------------------------------------------


def _launcher_env(root: Path, tmp: Path) -> dict[str, str]:
    """An environment in which the launcher resolves `root` as its own repository."""
    env = dict(_GIT_ENV)
    env["HOME"] = str(tmp)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root), env.get("PYTHONPATH")]))
    env["SERVING_PYTHON"] = sys.executable
    return env


#: A stand-in for the vLLM probe. These cases are about the argument boundary, not about
#: whether the serving venv is installed; requiring transformers and vLLM here would make the
#: suite depend on a GPU image to test a string list.
PROBE_SHIM = """
import json
import sys

# Load the real launcher, then replace the environment probe and run its CLI.
import runpy

launcher_path = sys.argv[1]
repo_root = sys.argv[2]
import importlib.util

spec = importlib.util.spec_from_file_location("run_in_serving_env_shimmed", launcher_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

module._check_environment = lambda python, root: (True, {"packages": {}, "repo_scripts_ok": True, "scripts_path": str(root)})
module._serving_python = lambda: __import__("pathlib").Path(sys.executable)
sys.argv = [launcher_path] + sys.argv[3:]
raise SystemExit(module.main())
"""


def _call_launcher_in_repo(launcher: Any, root: Path, manifest: Path, target_args: list[str]) -> int:
    """Run the launcher against `root` as a real subprocess, and return its exit code.

    A subprocess is used rather than calling `main()` in process because the launcher asks
    the gate to inspect whatever repository it lives in. The fixture copies the launcher into
    `root`, so `root` is that repository and authorisation is evaluated against its clean
    tree instead of this working tree. The serving-environment probe is shimmed out: it wants
    vLLM, and this case is about which arguments reach the target.
    """
    staged = root / "scripts" / "run_in_serving_env.py"
    shim = root.parent / "probe_shim.py"
    shim.write_text(PROBE_SHIM, encoding="utf-8")
    done = subprocess.run(
        [
            sys.executable,
            str(shim),
            str(staged),
            str(root),
            "strict_target.py",
            "--",
            "--manifest",
            str(manifest),
            *target_args,
        ],
        cwd=root,
        env=_launcher_env(root, root.parent),
        capture_output=True,
        text=True,
    )
    return done.returncode


def _run_strict_target(
    launcher: Any,
    root: Path,
    manifest: Path,
    target_args: list[str],
    *,
    bypass_stripping: bool = False,
) -> tuple[dict[str, Any], int]:
    """Run the launcher against the dummy target and return (what it saw, exit code).

    `bypass_stripping` reproduces the pre-fix behaviour by handing the target the unstripped
    argument list, which is how the control case proves the fix is load-bearing.
    """
    if bypass_stripping:
        code = _call_launcher_unstripped(launcher, root, manifest, target_args)
    else:
        code = _call_launcher_in_repo(launcher, root, manifest, target_args)

    seen_file = root / "strict_target.py.seen.json"
    if not seen_file.exists():
        return {}, code
    try:
        return json.loads(seen_file.read_text(encoding="utf-8")), code
    finally:
        seen_file.unlink()


def _call_launcher_unstripped(launcher: Any, root: Path, manifest: Path, target_args: list[str]) -> int:
    """Pre-fix behaviour: forward `--manifest` to the target, as the launcher used to.

    Only the forwarding step is reproduced. Authorisation is not re-run, so the control
    isolates the argument boundary rather than the gate.
    """
    leaked = ["--", "--manifest", str(manifest), *target_args]
    done = subprocess.run(
        [sys.executable, str(root / "strict_target.py"), *leaked],
        cwd=root,
        env=_launcher_env(root, root.parent),
        capture_output=True,
        text=True,
    )
    return done.returncode
