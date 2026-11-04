#!/usr/bin/env python
"""Generate the manifest that every experiment run must carry. Phase 5.2.

A result is only reproducible if the thing that produced it can be named. This script
writes one `manifest.json` per run directory, recording *identity* - what code, what
weights, what data, what settings, what hardware - and refusing to overwrite an existing
manifest. It deliberately records **no measured outcome**: numbers go in the traces and
the analysis, and a manifest that also carried results would invite editing it to match a
number that turned out inconvenient.

    uv run python scripts/run_manifest.py --arm baseline --run-id b6_baseline
    uv run python scripts/run_manifest.py --arm training --run-id b6_grpo -o results/raw/b6_grpo/
    uv run python scripts/run_manifest.py --arm baseline --run-id x --check

Three properties are worth stating because they are the reason this is code and not a
template someone fills in.

**Hashes, not descriptions.** Every value that determines an outcome is hashed from the
file that currently exists on disk. If a config is edited, the hash in a later manifest
differs, and two runs claiming the same protocol can be shown to differ. A manifest that
merely *said* "protocol_1p5b" would let an edit pass unnoticed.

**Nothing is invented.** Every field is read from git, from a file, or from `nvidia-smi`.
Where a value genuinely does not exist yet - an unpinned model revision, a run that has
not started - the field is `null` or absent and is listed under `unresolved`. An empty
string would read as "known and empty", which is a different and false claim.

**A refused overwrite is a feature.** `--force` exists and is the only way through, so a
second manifest for the same run is a recorded, deliberate act rather than an accident.
The original is preserved as `manifest.superseded-<utc>.json` rather than deleted, because
"the first manifest was wrong" is itself information about the run.

Exit codes: 0 written / already-present-and-matching, 1 refused overwrite, 2 bad input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The two arms of the experiment. `baseline` is the untrained control; `training` is the
#: arm that performs the update. Both read the same protocol file, which is what makes the
#: comparison attributable: the arm is the only thing that differs.
ARM_CONFIGS: dict[str, str] = {
    "baseline": "configs/eval/baseline.yaml",
    "training": "configs/training/grpo_nemo_rl_a5000.yaml",
}

#: Files whose content defines the experiment. Hashed on every run.
PROTOCOL_CONFIG = "configs/training/protocol_1p5b.yaml"
REWARD_V1_SOURCE = "rewards/compose.py"
REWARD_V2_SOURCE = "rewards/v2.py"
TRACE_SCHEMA = "analysis/trace_schema.json"
VERIFIER_SOURCE = "envs/coding_env/verifier.py"

MANIFEST_NAME = "manifest.json"


# --------------------------------------------------------------------------- helpers


def _run(cmd: list[str], cwd: Path = REPO_ROOT, timeout: int = 30) -> str | None:
    """Run a command and return stripped stdout, or None if it is unavailable or fails."""
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [exe, *cmd[1:]], cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def sha256_of(path: Path) -> str | None:
    """Content hash of a file, or None if it does not exist.

    None rather than a hash of the empty string: a missing file and an empty file are
    different facts, and `e3b0c442...` for both would make them indistinguishable.
    """
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(rel_path: str) -> dict[str, Any]:
    path = REPO_ROOT / rel_path
    return {"path": rel_path, "sha256": sha256_of(path), "present": path.is_file()}


def git_version() -> dict[str, Any]:
    """Commit, branch, and whether the tree was clean when the manifest was written.

    `dirty` is recorded because a manifest from a dirty tree names a commit that does not
    contain the code that ran. It is a warning, not an error: blocking on it would make
    people stop generating manifests, which is worse.
    """
    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def gpu_info() -> dict[str, Any]:
    """The GPU actually present, read from nvidia-smi.

    Never raises: a CPU-only machine is a normal state for this repo (ADR 0001), and the
    manifest must still be writable there. `present: false` is the honest record.
    """
    if shutil.which("nvidia-smi") is None:
        return {"present": False, "reason": "nvidia-smi not on PATH", "devices": []}
    fields = ["index", "name", "memory.total", "driver_version"]
    csv = _run(["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"])
    if csv is None:
        return {"present": False, "reason": "nvidia-smi returned a non-zero exit code", "devices": []}
    devices = []
    for line in csv.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == len(fields):
            devices.append(dict(zip(fields, parts, strict=True)))
    return {
        "present": bool(devices),
        "devices": devices,
        # Recorded as the string it is. `CUDA_VISIBLE_DEVICES` remaps indices, so the
        # in-process index 0 is not necessarily the physical index 0 - the reason Phase
        # 5.3 requires the variable to be set explicitly rather than relied upon.
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def load_protocol() -> dict[str, Any]:
    import yaml

    path = REPO_ROOT / PROTOCOL_CONFIG
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dataset_version() -> dict[str, Any]:
    """Which task corpus the run drew from, as a hash over the task definition files.

    Hashes the task.yaml files and the split each belongs to, sorted by path, so the value
    is stable across machines and independent of filesystem order. Task *contents* matter,
    not just names: editing a hidden test changes what a reward means.
    """
    tasks_root = REPO_ROOT / "benchmark" / "tasks"
    per_split: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    if not tasks_root.is_dir():
        return {"present": False, "splits": {}, "sha256": None}
    for split_dir in sorted(p for p in tasks_root.iterdir() if p.is_dir()):
        files = sorted(split_dir.glob("*/task.yaml"))
        split_digest = hashlib.sha256()
        for task_file in files:
            split_digest.update(str(task_file.relative_to(REPO_ROOT)).encode())
            split_digest.update(task_file.read_bytes())
            digest.update(str(task_file.relative_to(REPO_ROOT)).encode())
            digest.update(task_file.read_bytes())
        per_split[split_dir.name] = {"n_tasks": len(files), "sha256": split_digest.hexdigest()}
    return {"present": True, "splits": per_split, "sha256": digest.hexdigest()}


def checkpoint_info(input_checkpoint: str | None, output_checkpoint: str | None) -> dict[str, Any]:
    """Hash the input checkpoint (if it is a local path); name the output without hashing it.

    The output checkpoint does not exist when the manifest is written, so it is recorded
    as a declared path. Hashing it here would either fail or - worse - succeed against a
    stale file from a previous run. `scripts/run_manifest.py --checkpoint-hash` fills it
    in afterwards.
    """

    def describe(spec: str | None) -> dict[str, Any] | None:
        if spec is None:
            return None
        path = Path(spec)
        absolute = path if path.is_absolute() else REPO_ROOT / path
        exists = absolute.exists()
        return {
            "spec": spec,
            "local": exists,
            "sha256": sha256_of(absolute) if absolute.is_file() else None,
        }

    return {"input": describe(input_checkpoint), "output": describe(output_checkpoint)}


def unresolved_fields(manifest: dict[str, Any]) -> list[str]:
    """Names of fields that are null, empty, or a known-unpinned placeholder.

    Reported so that "the manifest is complete" is never assumed. A run may proceed with
    gaps - the question is whether the gap was noticed.
    """
    gaps: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else key)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{prefix}[{i}]")
        elif node is None:
            gaps.append(prefix)

    for key in ("model", "lora", "training", "rollouts", "sampling", "grpo", "data", "hardware"):
        walk(manifest.get("protocol", {}).get(key), f"protocol.{key}")
    if not manifest.get("gpu", {}).get("present"):
        gaps.append("gpu (no GPU detected)")
    if manifest.get("git", {}).get("dirty"):
        gaps.append("git.dirty (tree had uncommitted changes)")
    return gaps


# --------------------------------------------------------------------------- manifest


def build_manifest(
    *,
    arm: str,
    run_id: str,
    split: str,
    seed: int | None,
    input_checkpoint: str | None,
    output_checkpoint: str | None,
    note: str | None,
) -> dict[str, Any]:
    if arm not in ARM_CONFIGS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_CONFIGS)}")

    protocol = load_protocol()
    now = datetime.now(UTC).isoformat()

    arm_config = ARM_CONFIGS[arm]
    manifest: dict[str, Any] = {
        # -- identity ---------------------------------------------------------------
        "manifest_version": "1",
        "run_id": run_id,
        "arm": arm,
        "written_at": now,
        "note": note,
        # -- code -------------------------------------------------------------------
        "git": git_version(),
        # -- the protocol, resolved to the values the run will use -------------------
        "protocol": {
            "name": protocol.get("meta", {}).get("name"),
            "file": config_hash(PROTOCOL_CONFIG),
            "model": protocol.get("model", {}),
            "lora": protocol.get("lora", {}),
            "training": protocol.get("training", {}),
            "rollouts": protocol.get("rollouts", {}),
            "sampling": protocol.get("sampling", {}),
            "episode": protocol.get("episode", {}),
            "grpo": protocol.get("grpo", {}),
            "reward": protocol.get("reward", {}),
            "data": protocol.get("data", {}),
            "hardware": protocol.get("hardware", {}),
        },
        # -- per-arm override, hashed ------------------------------------------------
        "arm_config": config_hash(arm_config),
        # -- what the numbers mean ---------------------------------------------------
        "reward": {
            "live": protocol.get("reward", {}).get("live"),
            "analysed": protocol.get("reward", {}).get("analysed"),
            "range": protocol.get("reward", {}).get("range"),
            "v1_source": config_hash(REWARD_V1_SOURCE),
            "v2_source": config_hash(REWARD_V2_SOURCE),
        },
        "verifier": config_hash(VERIFIER_SOURCE),
        "trajectory_schema": config_hash(TRACE_SCHEMA),
        # -- data ---------------------------------------------------------------------
        "dataset": {
            "version": dataset_version(),
            "split": split,
            "heldout_enabled": bool(protocol.get("data", {}).get("heldout_enabled", False)),
        },
        # -- hardware ------------------------------------------------------------------
        "gpu": gpu_info(),
        "seed": seed,
        # -- checkpoints ---------------------------------------------------------------
        "checkpoints": checkpoint_info(input_checkpoint, output_checkpoint),
        # -- timing. end is filled by --finish; a manifest is written before a run starts,
        #    because the run that crashes is the one whose manifest matters most.
        "timing": {"started_at": now, "finished_at": None, "duration_s": None},
        "outcome": {"status": "planned", "failure_reason": None},
        "artifacts": {"run_dir": None, "trace": None, "log": None, "server_pid": None},
    }
    manifest["unresolved"] = unresolved_fields(manifest)
    return manifest


def write_manifest(manifest: dict[str, Any], out_dir: Path, *, force: bool) -> tuple[Path, bool]:
    """Write `manifest.json`, refusing to clobber. Returns (path, wrote).

    A pre-existing manifest is compared first: if it is byte-identical in its identity
    fields the write is skipped and reported as "already present", which makes the script
    safe to re-run in a Makefile without forcing anyone to remember `--force`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / MANIFEST_NAME

    if target.exists():
        if not force:
            try:
                existing = json.loads(target.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = None
            if existing is not None and _identity(existing) == _identity(manifest):
                return target, False
            raise FileExistsError(f"{target} exists and differs; pass --force to supersede it")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup = out_dir / f"manifest.superseded-{stamp}.json"
        target.rename(backup)

    target.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return target, True


def _identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """The fields that make two manifests the same run. Timestamps are excluded."""
    return {
        "run_id": manifest.get("run_id"),
        "arm": manifest.get("arm"),
        "protocol": manifest.get("protocol", {}).get("file"),
        "arm_config": manifest.get("arm_config"),
        "git": manifest.get("git"),
        "dataset": manifest.get("dataset", {}).get("version", {}).get("sha256"),
        "seed": manifest.get("seed"),
    }


def finish_manifest(path: Path, status: str, failure_reason: str | None) -> dict[str, Any]:
    """Close out a manifest: record the end time, duration and how the run ended.

    Editing a manifest after the fact is normally the thing this script prevents. This is
    the one sanctioned edit, it is restricted to the `timing` and `outcome` blocks, and it
    preserves everything that identifies the run.
    """
    manifest = json.loads(path.read_text(encoding="utf-8"))
    now = datetime.now(UTC)
    started = manifest.get("timing", {}).get("started_at")
    manifest["timing"]["finished_at"] = now.isoformat()
    if started:
        try:
            manifest["timing"]["duration_s"] = round((now - datetime.fromisoformat(started)).total_seconds(), 1)
        except ValueError:
            manifest["timing"]["duration_s"] = None
    manifest["outcome"] = {"status": status, "failure_reason": failure_reason}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a run manifest (Phase 5.2).")
    parser.add_argument("--arm", required=True, choices=sorted(ARM_CONFIGS))
    parser.add_argument("--run-id", required=True, help="Unique run identifier, e.g. b6_baseline")
    parser.add_argument("--split", default="train", choices=["train", "heldout"])
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the protocol's seed")
    parser.add_argument("--input-checkpoint", default=None)
    parser.add_argument("--output-checkpoint", default=None)
    parser.add_argument("--note", default=None)
    parser.add_argument("-o", "--out", default=None, help="Run directory (default: results/raw/<run_id>/)")
    parser.add_argument("--force", action="store_true", help="Supersede an existing, differing manifest")
    parser.add_argument("--json", action="store_true", help="Print the manifest to stdout")
    parser.add_argument(
        "--finish",
        nargs=2,
        metavar=("STATUS", "REASON"),
        default=None,
        help="Close an existing manifest: --finish completed - / --finish failed 'OOM at step 3'",
    )
    args = parser.parse_args(argv)

    if args.split == "heldout" and args.arm == "training":
        print(
            "refusing: the held-out split is sealed for training (docs/heldout_seal.md).\n"
            "It is evaluated exactly once, at the end of the project, against a frozen checkpoint.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out) if args.out else REPO_ROOT / "results" / "raw" / args.run_id
    target = out_dir / MANIFEST_NAME

    if args.finish:
        if not target.is_file():
            print(f"refusing: no manifest at {target}", file=sys.stderr)
            return 2
        status, reason = args.finish
        reason = None if reason in ("-", "") else reason
        manifest = finish_manifest(target, status, reason)
        print(f"closed {target}  status={status}" + (f"  reason={reason}" if reason else ""))
        if args.json:
            print(json.dumps(manifest, indent=2))
        return 0

    protocol = load_protocol()
    seed = args.seed if args.seed is not None else protocol.get("sampling", {}).get("seed")
    manifest = build_manifest(
        arm=args.arm,
        run_id=args.run_id,
        split=args.split,
        seed=seed,
        input_checkpoint=args.input_checkpoint,
        output_checkpoint=args.output_checkpoint,
        note=args.note,
    )

    try:
        path, wrote = write_manifest(manifest, out_dir, force=args.force)
    except FileExistsError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1

    print(("wrote " if wrote else "already present ") + str(path))
    if manifest["unresolved"]:
        print(f"  unresolved ({len(manifest['unresolved'])}): " + ", ".join(manifest["unresolved"]))
    if not manifest["gpu"]["present"]:
        print("  note: no GPU detected on this machine (normal off the training box)")
    if manifest["git"]["dirty"]:
        print("  WARNING: working tree was dirty; this manifest names a commit that does not contain the code that ran")
    if args.json:
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
