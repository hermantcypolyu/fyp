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
import copy
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

#: The arms of the experiment. `baseline` is the untrained control; `training` is the arm
#: that performs the update; `pilot` is a rollout-only smoke run that updates nothing.
#: All three read the same formal protocol file, which is what makes the comparison
#: attributable: the arm is the only thing that differs. `pilot` additionally carries its
#: own overlay, which is recorded and hashed separately from the formal protocol - a pilot
#: whose overlay was not hashed could be edited between the pilot and the claim about it.
ARM_CONFIGS: dict[str, str] = {
    "baseline": "configs/eval/baseline.yaml",
    "training": "configs/training/grpo_nemo_rl_a5000.yaml",
    "pilot": "configs/training/pilot_rollouts_1p5b.yaml",
}

#: Runs that never perform a parameter update, whatever else they do.
NON_UPDATING_ARMS: frozenset[str] = frozenset({"pilot"})

#: Files whose content defines the experiment. Hashed on every run.
PROTOCOL_CONFIG = "configs/training/protocol_1p5b.yaml"
#: The rollout-only overlay. Hashed on every run so the pilot cannot be described by a file
#: that has since changed.
PILOT_CONFIG = "configs/training/pilot_rollouts_1p5b.yaml"
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


def load_pilot() -> dict[str, Any]:
    """The rollout-only overlay, or `{}` when it is absent.

    Absent is not an error: only the `pilot` arm reads it, and a missing file then shows up
    as a null hash in the manifest and as a gap in `unresolved` rather than a crash.
    """
    import yaml

    path = REPO_ROOT / PILOT_CONFIG
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


def _scan_tree(root: Path) -> tuple[str | None, int, int] | None:
    """Walk `root` once and return `(tree_sha256, file_count, total_bytes)`.

    Returns `None` when the tree holds no files. See `sha256_of_tree` for the hashing
    contract; this helper exists so the digest, the file count and the byte total all come
    from the *same* walk rather than three separate traversals that could disagree.
    """
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        return None
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        contents = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
        total_bytes += len(contents)
    return digest.hexdigest(), len(files), total_bytes


def sha256_of_tree(root: Path) -> str | None:
    """Hash a directory of checkpoint files deterministically, or `None` if it holds none.

    A checkpoint is usually a directory (`model.safetensors`, `config.json`, adapter
    weights), and hashing only single files meant a completed run's weights went
    unrecorded. This walks the tree and hashes it the way `dataset_version` hashes the task
    corpus: relative paths sorted, each path and its bytes fed in, nothing else.

    Deliberately excluded, because none of them describe the weights and including them
    would make two identical checkpoints hash differently:

      * mtimes, atimes and ctime - a copy is byte-identical but has a new mtime;
      * permissions and ownership - a `chmod` is not a change to the model;
      * directory entries themselves - only the relative path *text* is hashed, so an empty
        subdirectory leaves no trace and `a/b.bin` hashes the same wherever `root` is;
      * filesystem iteration order - `sorted` supplies the order, so the same content
        written in any order gives the same digest.

    The empty directory is the one case that needs a stated answer. A directory containing
    no files returns `None`, not the hash of nothing, because an empty checkpoint directory
    means the run produced no weights and that must be visible rather than recorded as a
    successful hash of emptiness.

    This is the digest-only view of `_scan_tree`; callers that also want the file count and
    byte total should use that helper so both numbers come from one walk.
    """
    scanned = _scan_tree(root)
    return None if scanned is None else scanned[0]


def _describe_checkpoint(spec: str | None) -> dict[str, Any] | None:
    """Describe one checkpoint path: absence, a single file, or a whole directory.

    A directory record carries, in one traversal:
      * `kind: "directory"` - distinguishes weights-on-disk from a single file;
      * `sha256` / `tree_sha256` - the deterministic tree digest from `_scan_tree`;
      * `file_count`, `total_bytes` - the shape of the checkpoint, so a truncated or
        partially-written save is visible without re-hashing;
      * `reason: "empty_directory"` - set only when the directory holds no files, in which
        case the digest fields are absent rather than `null`-hashed.
    """
    if spec is None:
        return None
    path = Path(spec)
    absolute = path if path.is_absolute() else REPO_ROOT / path
    if not absolute.exists():
        return {"spec": spec, "local": False, "kind": "absent", "sha256": None}
    if absolute.is_file():
        return {"spec": spec, "local": True, "kind": "file", "sha256": sha256_of(absolute)}
    scanned = _scan_tree(absolute)
    if scanned is None:
        return {
            "spec": spec,
            "local": True,
            "kind": "directory",
            "sha256": None,
            "reason": "empty_directory",
        }
    tree_hash, file_count, total_bytes = scanned
    # `sha256` and `tree_sha256` are the same digest under two names: `sha256` keeps every
    # record (file, directory, dataset) readable at one key, `tree_sha256` names what it is.
    return {
        "spec": spec,
        "local": True,
        "kind": "directory",
        "sha256": tree_hash,
        "tree_sha256": tree_hash,
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def checkpoint_info(input_checkpoint: str | None, output_checkpoint: str | None) -> dict[str, Any]:
    """Hash the input checkpoint (if it is a local path); name the output without hashing it.

    The output checkpoint does not exist when the manifest is written, so it is recorded
    as a declared path. Hashing it here would either fail or - worse - succeed against a
    stale file from a previous run. `resolve_output_checkpoint` fills it in at `--finish`,
    which is the first moment the file can legitimately exist.

    An input *may* be hashed eagerly, and a directory input is hashed as a tree, because an
    input that already exists is unambiguously the thing the run will read.
    """
    return {
        "input": _describe_checkpoint(input_checkpoint),
        "output": _describe_checkpoint(output_checkpoint),
    }


def resolve_output_checkpoint(manifest: dict[str, Any]) -> dict[str, Any]:
    """Re-inspect a declared output checkpoint, at close-out, against the filesystem.

    Before a run the output is a promise: `local: false`, `sha256: null`. After a run it
    is either a fact or a failure, and the difference matters:

      * a directory of weights -> `local: true`, `kind: directory`, and its tree hash;
      * a single weight file   -> `local: true`, `kind: file`, and its file hash;
      * it does not exist      -> `local: false`, `sha256: null`, and
        `reason: missing_at_close`, so a successful run that produced no weights is
        visible rather than implied by an optimistic `local: true`.

    What this must never do is invent a hash. A directory that exists but holds no file
    (an empty checkpoint dir, or a partially written one) reports `local: true` with a
    null hash and `reason: empty_directory`, because "the path exists" and "there are
    weights there" are different claims and only the second one makes a result
    reproducible.
    """
    output = (manifest.get("checkpoints") or {}).get("output")
    if output is None:
        return manifest
    spec = output.get("spec")
    if spec is None:
        return manifest

    resolved = _describe_checkpoint(spec)
    if resolved is None:
        return manifest
    resolved.pop("spec", None)
    output.update(resolved)
    if not resolved["local"]:
        output["reason"] = "missing_at_close"
    return manifest
    return manifest


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
    # A pilot run whose overlay was edited away, or whose tasks list is still unset, is a
    # gap worth naming rather than a detail hidden inside the hashed file.
    if manifest.get("pilot", {}).get("applies"):
        if manifest["pilot"].get("file", {}).get("sha256") is None:
            gaps.append("pilot.file (overlay missing)")
        if manifest["pilot"].get("tasks") is None:
            gaps.append("pilot.tasks (not yet selected)")
    if not manifest.get("gpu", {}).get("present"):
        gaps.append("gpu (no GPU detected)")
    if manifest.get("git", {}).get("dirty"):
        gaps.append("git.dirty (tree had uncommitted changes)")
    return gaps


# --------------------------------------------------------------------------- manifest


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Merge `overlay` onto `base`, returning the merged document and per-key provenance.

    Provenance is recorded as a dotted path -> source name for every key the overlay
    supplied. It exists because a manifest that lists `rollouts_per_task: 8` is making a
    claim about the run, and the reader has to be able to tell whether 8 came from the
    formal protocol, the pilot overlay, or a command-line override. Without it, an overlay
    default silently impersonates a protocol value.
    """
    merged = copy.deepcopy(base)
    provenance: dict[str, str] = {}

    def _walk(node: Any, prefix: str) -> None:
        for key, value in node.items():
            dotted = f"{prefix}{key}"
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                _walk(value, f"{dotted}.")
            else:
                merged[key] = copy.deepcopy(value)
                provenance[dotted] = "overlay"

    _walk(overlay, "")
    return merged, provenance


def effective_settings(
    protocol: dict[str, Any],
    pilot: dict[str, Any],
    *,
    arm: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The settings the run actually used, with hashes and provenance.

    The base protocol is always hashed. The overlay hash is unconditional for the same
    reason `pilot.file` is: it answers "which file existed and what was in it", independent
    of whether this arm applied it. The merged hash is over the merged document *after*
    overrides, so it fingerprints the run rather than the intent.
    """
    merged, provenance = _deep_merge(protocol, pilot)

    for name, value in (overrides or {}).items():
        merged[name] = value
        provenance[name] = "cli override"

    rollouts = merged.get("rollouts") or {}

    for key in ("rollouts_per_task", "tasks"):
        if f"rollouts.{key}" in provenance:
            provenance[f"rollouts.{key}"] = (
                f"overlay ({PILOT_CONFIG})" if arm in NON_UPDATING_ARMS else "overlay (not applied to this arm)"
            )

    # Canonical JSON, sorted, so the hash depends on the settings and not on YAML ordering.
    # `default=str` covers values YAML parsed into non-JSON scalars (dates in the protocol
    # header), which are still settings that must be part of the fingerprint.
    encoded = json.dumps(merged, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")

    return {
        # -- what this arm actually resolved to ---------------------------------
        "rollouts_per_task": rollouts.get("rollouts_per_task"),
        "tasks": rollouts.get("tasks"),
        "concurrency": (merged.get("hardware") or {}).get("concurrency"),
        # -- hashes over the three documents ------------------------------------
        "hashes": {
            "protocol": config_hash(PROTOCOL_CONFIG),
            "overlay": config_hash(PILOT_CONFIG),
            "merged": {"sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded)},
        },
        # -- where each value came from -----------------------------------------
        "provenance": dict(sorted(provenance.items())),
    }


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
    pilot = load_pilot()
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
        # -- the resolved settings, as opposed to the files that were read ----------
        #    `protocol` and `pilot` above say what each file contained. This block says
        #    what the run actually used after the overlay was merged, and which layer
        #    supplied each value. Phase 6E showed why this is not redundant: its manifest
        #    reported `rollouts_per_task: 8` and `tasks: null` from the overlay defaults
        #    while the run executed 4 rollouts over 1 task, and nothing in the file could
        #    distinguish "the overlay said 8" from "the run used 8".
        "effective": effective_settings(protocol, pilot, arm=arm),
        # -- the rollout-only overlay, recorded so a pilot run names the file that
        #    defined it rather than only the formal protocol it did not execute.
        #    The file hash is unconditional: it answers "did the overlay exist and has
        #    anyone edited it", which is true whether or not this arm used it. The run's
        #    settings are null on other arms, because a baseline manifest asserting
        #    `training_iterations: 0` from a file it never read would be a false claim
        #    dressed up as completeness. -----------------------------------------------
        "pilot": {
            "applies": arm in NON_UPDATING_ARMS,
            "file": config_hash(PILOT_CONFIG),
            "name": pilot.get("meta", {}).get("name") if arm in NON_UPDATING_ARMS else None,
            "formal_protocol": (pilot.get("meta", {}).get("formal_protocol") if arm in NON_UPDATING_ARMS else None),
            "performs_parameter_update": (
                (pilot.get("purpose") or {}).get("performs_parameter_update") if arm in NON_UPDATING_ARMS else None
            ),
            "training_iterations": (
                (pilot.get("training") or {}).get("training_iterations") if arm in NON_UPDATING_ARMS else None
            ),
            "optimizer_enabled": (
                (pilot.get("training") or {}).get("optimizer_enabled") if arm in NON_UPDATING_ARMS else None
            ),
            "expect_output_checkpoint": (
                (pilot.get("training") or {}).get("expect_output_checkpoint") if arm in NON_UPDATING_ARMS else None
            ),
            "rollouts_per_task": (
                (pilot.get("rollouts") or {}).get("rollouts_per_task") if arm in NON_UPDATING_ARMS else None
            ),
            "tasks": (pilot.get("rollouts") or {}).get("tasks") if arm in NON_UPDATING_ARMS else None,
            "concurrency": (pilot.get("hardware") or {}).get("concurrency") if arm in NON_UPDATING_ARMS else None,
            "heldout_enabled": (pilot.get("data") or {}).get("heldout_enabled") if arm in NON_UPDATING_ARMS else None,
            "reward_offline_only": (
                (pilot.get("reward") or {}).get("offline_only") if arm in NON_UPDATING_ARMS else None
            ),
            "reward_v2_computed_inline": (
                (pilot.get("reward") or {}).get("reward_v2_computed_inline") if arm in NON_UPDATING_ARMS else None
            ),
        },
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
    the one sanctioned edit, it is restricted to the `timing`, `outcome` and
    `checkpoints.output` blocks, and it preserves everything that identifies the run.

    The output checkpoint is re-inspected here because this is the first moment it can
    legitimately exist. Before close-out it is a declared path with `sha256: null`; after,
    it either has a real hash or an explicit reason why it has none. It stays part of
    `_identity`-preserving state: only `local`, `sha256` and `reason` move.
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
    resolve_output_checkpoint(manifest)
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

    if args.split == "heldout" and args.arm in NON_UPDATING_ARMS | {"training"}:
        print(
            "refusing: the held-out split is sealed for training (docs/heldout_seal.md).\n"
            "It is evaluated exactly once, at the end of the project, against a frozen checkpoint.\n"
            "A rollout-only pilot does not create that checkpoint, so it still cannot open the seal.",
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
