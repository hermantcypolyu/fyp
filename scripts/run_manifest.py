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
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

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


def _untracked_paths() -> list[str] | None:
    """Paths git knows about that are not tracked, or None when git is unavailable.

    `--exclude-standard` applies `.gitignore`, so ignored files (everything under
    `results/`) are absent - which is the point. An untracked *source* or *config* file is a
    different animal from an untracked run artefact: the former means the code that ran
    may not exist in the commit the manifest names, and the latter is a by-product.
    """
    listing = _run(["git", "ls-files", "--others", "--exclude-standard"])
    if listing is None:
        return None
    return [line for line in listing.splitlines() if line.strip()]


def git_version() -> dict[str, Any]:
    """Commit, branch, and what state the working copy was in when the manifest was written.

    Three facts are recorded rather than one, because they mean different things to a
    reader deciding whether a run may start:

    * `dirty` - tracked files differ from HEAD. The manifest names a commit that does not
      contain the code that ran. This *is* an execution blocker when the manifest claims to
      authorise a run; see `execution_claim`.
    * `untracked` - files git does not track and `.gitignore` does not exclude. A stray
      untracked config can silently change what a run picks up, so it is reported
      separately rather than folded into `dirty`, where a `results/` artefact would look
      indistinguishable from an uncommitted source edit.
    * `untracked_ignored_count` - the number of ignored files that exist (mostly under
      `results/`). Recorded so nobody has to guess whether an empty untracked list meant
      "nothing there" or "nothing that counts". These never influence execution.
    """
    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    # `--untracked-files=no` on purpose: `dirty` answers "did tracked content change?", and
    # an untracked file cannot change what a commit contains.
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    untracked = _untracked_paths()
    ignored = _run(["git", "ls-files", "--others", "--ignored", "--exclude-standard"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
        "untracked": untracked,
        "untracked_ignored_count": None if ignored is None else len([ln for ln in ignored.splitlines() if ln.strip()]),
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
    #
    # The overlay placeholder is *not* the executed identity. `pilot_rollouts_1p5b.yaml`
    # ships `rollouts.tasks: null` on purpose ("not decided, not invented"), but a run that
    # executes a materialised eval config never reads that file - the collector loads one
    # config and resolves no `extends:`. So the gap is only blocking when the executed block
    # is absent or itself incomplete; when the executed settings name a real task list, the
    # overlay placeholder is recorded as an overlay value and stops blocking. Naming it
    # anyway would report a decision the run did not need to make, alongside the one it did.
    if manifest.get("pilot", {}).get("applies"):
        if manifest["pilot"].get("file", {}).get("sha256") is None:
            gaps.append("pilot.file (overlay missing)")
        executed = manifest.get("executed") or {}
        executed_tasks = (executed.get("settings") or {}).get("tasks")
        # Read the nested value: `pilot.values` is where overlay values live, and the
        # flattened key this used to look at no longer exists.
        overlay_tasks = (manifest["pilot"].get("values") or {}).get("tasks")
        if overlay_tasks is None and not executed_tasks:
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

    def _walk(source: dict[str, Any], target: dict[str, Any], prefix: str) -> None:
        # `target` is threaded through explicitly rather than reaching for an enclosing
        # `merged`: the first version of this wrote every nested key back onto the ROOT
        # document, so `hardware.concurrency: 1` landed as a top-level `concurrency` and
        # `hardware.concurrency` stayed absent. The merged hash still looked plausible,
        # which is what made it dangerous - the whole point of the hash is to fingerprint
        # the effective settings, and it was fingerprinting a document with 50-odd keys
        # in the wrong place.
        for key, value in source.items():
            dotted = f"{prefix}{key}"
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                _walk(value, target[key], f"{dotted}.")
            else:
                target[key] = copy.deepcopy(value)
                provenance[dotted] = "overlay"

    _walk(overlay, merged, "")
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


# Keys whose *absence* is still meaningful, but which may legitimately be omitted from a
# materialised config. `model.adapter` is the only one: the checked-in baseline writes
# `adapter: null` to say "base model", but a merge that simply never mentions it states the
# same thing, and refusing to call that materialised would be pedantry that hides real gaps.
#
# The three update-related keys are optional for a different reason: they are properties of
# the *pilot overlay*, and a sample-and-collect eval config has no reason to restate them.
# Their absence means "this config says nothing about updating", which is not the same as
# "this config updates" - so they are read as None rather than as a missing key, and the
# executability gate treats None as "no update", never as "unknown update".
OPTIONAL_IN_EXECUTED = frozenset(
    {
        "model.adapter",
        "run.optimizer_enabled",
        "run.training_iterations",
        "run.expect_output_checkpoint",
    }
)

#: The `settings` names that a null value is allowed to carry. `adapter` is the setting
#: name for the dotted `model.adapter` above; the two spellings are kept apart because the
#: dotted form addresses the config file and this one addresses the flattened record. The
#: update-related names join it for the reason given above: their absence is an answer.
OPTIONAL_IN_EXECUTED_NAMES: frozenset[str] = frozenset(
    {"adapter", "optimizer_enabled", "training_iterations", "expect_output_checkpoint"}
)

# --- artifact path mapping -------------------------------------------------------------
#
# One table, read by everything that relates a collector output key to a manifest field.
# Before this existed the relation was spelled out in three places (the extraction loop,
# the presence check, and the report), each with its own idea of which name meant what -
# which is how `output.private` came to be checked against `artifacts.private` and reported
# missing while `artifacts.private_trace` held the value all along.
#
# The table is the *only* place the correspondence is written down. `EXECUTED_OUTPUT_MAP`
# maps a collector `output.<key>` onto the manifest field it populates; `run_dir` and
# `summary` are derived rather than read, because the collector derives them too.


class OutputMapping(NamedTuple):
    """Where one collector output key lands in the manifest.

    `field` is the `executed.artifacts` name the raw path is recorded under. `derive`
    names the transformation applied to that path, because two of the five destinations
    are not the configured path itself:

      * ``run_dir`` is the *directory holding* the public rollout file;
      * ``summary`` is the collector's `collection_summary.json` written *beside* it.

    Making the derivation part of the table (rather than a special case in a loop) means a
    reader can see, in one place, that every required output has exactly one destination.
    """

    key: str
    field: str
    derive: str | None = None
    required: bool = False


#: The authoritative mapping. Order is the order a reader expects to meet these paths.
#: `output.rollouts` feeds two fields: the trace itself and the directory it lives in, which
#: is why `run_dir` is a `derive` entry on the same key rather than a separate config key
#: (there is no `output.run_dir`, and inventing one would let the two disagree).
OUTPUT_MAP: tuple[OutputMapping, ...] = (
    OutputMapping(key="rollouts", field="trace", required=True),
    OutputMapping(key="rollouts", field="run_dir", derive="parent", required=True),
    OutputMapping(key="private", field="private_trace", required=True),
    OutputMapping(key="failures", field="failures", required=True),
    OutputMapping(key="rollouts", field="summary", derive="sibling:collection_summary.json", required=True),
)

#: `output.<key>` -> the `executed.artifacts` field it fills, for the one-to-one entries.
EXECUTED_OUTPUT_ALIASES: dict[str, str] = {
    mapping.key: mapping.field for mapping in OUTPUT_MAP if mapping.derive is None
}

#: The `output.*` keys a materialised run config must name. Derived from the map so this
#: list cannot drift from the mapping it describes.
EXECUTED_REQUIRED_OUTPUTS: tuple[str, ...] = tuple(
    dict.fromkeys(mapping.key for mapping in OUTPUT_MAP if mapping.required)
)

#: Every `executed.artifacts` field the map can populate. Used by the presence check.
EXECUTED_ARTIFACT_FIELDS: tuple[str, ...] = tuple(dict.fromkeys(mapping.field for mapping in OUTPUT_MAP))


#: The manifest artifact fields that exist, for validating the map and rejecting typos.
KNOWN_ARTIFACT_FIELDS: frozenset[str] = frozenset(
    {"run_dir", "trace", "private_trace", "failures", "summary", "log", "server_pid"}
)


def _check_output_map() -> None:
    """Reject a mapping table that would let two outputs fight over one field.

    Called at import time. A duplicate field would silently make one requirement
    unverifiable - the second write would always win and the first output's absence would
    never be noticed - so it is refused loudly rather than tolerated.
    """
    fields = [mapping.field for mapping in OUTPUT_MAP]
    duplicates = sorted({field for field in fields if fields.count(field) > 1})
    if duplicates:
        raise ValueError(f"OUTPUT_MAP assigns more than one output to {duplicates}")
    keys = {mapping.key for mapping in OUTPUT_MAP}
    unknown = sorted(keys - set(EXECUTED_REQUIRED_OUTPUTS) - {"rollouts", "private", "failures"})
    if unknown:
        raise ValueError(f"OUTPUT_MAP names output keys the collector never opens: {unknown}")
    for mapping in OUTPUT_MAP:
        if mapping.field not in KNOWN_ARTIFACT_FIELDS:
            raise ValueError(f"OUTPUT_MAP writes {mapping.field!r}, which is not a manifest artifact field")


_check_output_map()


def _apply_derive(value: str, derive: str) -> str:
    """Transform a configured path into the manifest field it feeds.

    ``parent`` is the containing directory; ``sibling:<name>`` replaces the leaf with a
    file of that name in the same directory. Both go through `PurePosixPath` so a config
    written with forward slashes normalises the same way on any platform, and a path is
    never rewritten to something the collector would not actually open.
    """
    path = PurePosixPath(value)
    if derive == "parent":
        return str(path.parent)
    if derive.startswith("sibling:"):
        return str(path.parent / derive.split(":", 1)[1])
    raise ValueError(f"unknown derive rule {derive!r}")


def map_outputs(raw_outputs: dict[str, Any]) -> dict[str, str | None]:
    """Turn a materialised config's `output` block into manifest artifact fields.

    Returns every field the map can fill, with `None` where the config named no path. The
    caller decides whether a `None` is disqualifying; this function does not guess, because
    a pilot manifest legitimately has no executed config to read at all.
    """
    mapped: dict[str, str | None] = {}
    for mapping in OUTPUT_MAP:
        raw = raw_outputs.get(mapping.key)
        if not isinstance(raw, str):
            mapped[mapping.field] = None
            continue
        mapped[mapping.field] = _apply_derive(raw, mapping.derive) if mapping.derive else raw
    return mapped


def executed_config(rel_path: str) -> dict[str, Any]:
    """Fingerprint the config a diagnostic run actually executes, and read its settings.

    Why this exists. `effective_settings` above describes the *training* pair
    (`protocol_1p5b.yaml` + `pilot_rollouts_1p5b.yaml`), which is what a GRPO arm runs.
    A Phase 6 diagnostic smoke runs something else entirely: `configs/eval/baseline_phase6.yaml`
    materialised with a smoke overlay. Phase 6E hit exactly this and its manifest could say
    `rollouts_per_task: 8` while the collector ran 4, because nothing in the manifest had
    ever read the config the collector was handed. The numbers were individually true and
    jointly misleading, which is the failure mode the `effective` block was added to end.

    The collector loads ONE file with `BaselineConfig.load` and resolves no `extends:` at
    all - an overlay carrying only its differences is not runnable, it must be materialised
    first. So the value recorded here is the materialised file, hashed byte-for-byte, plus
    the handful of settings a reader needs to know what the run was. A materialised file
    that is missing any of them raises here rather than at run start.
    """
    import yaml

    path = REPO_ROOT / rel_path
    record: dict[str, Any] = {
        "path": rel_path,
        "sha256": None,
        "present": path.is_file(),
        "materialised": False,
    }
    if not path.is_file():
        record["error"] = "file not found; the manifest cannot claim settings it never read"
        return record

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record["sha256"] = digest
    record["bytes"] = path.stat().st_size

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    def _dig(dotted: str) -> Any:
        node: Any = raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                # A missing key here means the file is not the materialised config it is
                # claimed to be. Record the gap explicitly instead of printing a null that
                # reads like a deliberate choice.
                raise KeyError(dotted)
            node = node[part]
        return node

    # The run's *intended* artifact paths, read from the same file the collector loads.
    #
    # These are separate from `build_manifest`'s `artifacts` block on purpose: this one is
    # the intent recorded by the config (what the run will write), that one is the observed
    # reality after execution (what was written, plus the server PID and log). Recording the
    # intent here is what stops a planned run from showing `artifacts.run_dir: null` and
    # being indistinguishable from a manifest that never named a run directory at all.
    #
    # The correspondence itself lives in `OUTPUT_MAP`, not here. This block only decides how
    # to read a config that may be missing a key; the mapping decides where each key lands.
    raw_output = raw.get("output")
    artifacts = map_outputs(raw_output if isinstance(raw_output, dict) else {})
    record["artifacts"] = artifacts
    record["artifacts_log"] = (
        "these paths are the *intended* outputs declared by the config; the top-level "
        "`artifacts` block records what the run actually produced"
    )

    wanted = {
        "run_id": "output.run_id",
        "tasks": "run.task_subset",
        "rollouts_per_task": "run.rollouts_per_task",
        "min_valid_rollouts": "run.min_valid_rollouts",
        "min_distinct_rewards": "run.min_distinct_rewards",
        "diagnostic_only": "run.diagnostic_only",
        "model_name": "model.name",
        "model_revision": "model.revision",
        "adapter": "model.adapter",
        "temperature": "sampling.temperature",
        "top_p": "sampling.top_p",
        "seed": "sampling.seed",
        "max_tokens_per_turn": "sampling.max_tokens_per_turn",
        "max_turns": "episode.max_turns",
        "max_tool_calls": "episode.max_tool_calls",
        "timeout_s_per_rollout": "episode.timeout_s_per_rollout",
        "split": "tasks.split",
        "heldout_enabled": "tasks.heldout_enabled",
        # The three facts that say whether this is a rollout-only run. An eval config may
        # legitimately omit them - they are properties of the *pilot overlay*, not of a
        # sample-and-collect config - so they are read with `_dig_optional` below rather
        # than counted as missing keys. A run that silently accepted `optimizer_enabled:
        # true` would update weights under a manifest claiming it did not.
        "optimizer_enabled": "run.optimizer_enabled",
        "training_iterations": "run.training_iterations",
        "expect_output_checkpoint": "run.expect_output_checkpoint",
        "reward_live": "reward.live",
        "reward_analysed": "reward.analysed",
        "reward_v2_computed_inline": "reward.reward_v2_computed_inline",
        "served_model_name": "server.served_model_name",
        "cuda_visible_devices": "server.cuda_visible_devices",
        "max_model_len": "context.max_model_len",
        "reserved_completion_tokens": "context.reserved_completion_tokens",
        "safety_margin_tokens": "context.safety_margin_tokens",
    }
    settings: dict[str, Any] = {}
    missing: list[str] = []
    for name, dotted in wanted.items():
        try:
            settings[name] = _dig(dotted)
        except KeyError:
            settings[name] = None
            # `model.adapter: null` is a legitimate statement - "run the base model" - and
            # the checked-in baseline says it explicitly. Its *absence* is not, so it stays
            # in `wanted` for the value but is exempt from the materialisation check.
            if dotted not in OPTIONAL_IN_EXECUTED:
                missing.append(dotted)

    # An overlay is not materialised, so it cannot drive a run. Say that plainly rather
    # than emitting a settings block full of nulls that look like deliberate choices.
    #
    # The artifact paths are part of the same check. A file that defines the model and the
    # sampler but not where the run writes its evidence is not a run config: the collector
    # would raise on the first `output_path()` call, after the server was already up.
    #
    # A required output is satisfied when *any* field it maps to is populated, which is why
    # this reads the map rather than re-deriving the field name. Reading the wrong field
    # is exactly how `output.private` was once reported missing while `private_trace` held
    # its value.
    missing_outputs = sorted(
        f"output.{key}"
        for key in EXECUTED_REQUIRED_OUTPUTS
        if not any(artifacts.get(mapping.field) for mapping in OUTPUT_MAP if mapping.key == key)
    )
    for dotted in missing_outputs:
        if dotted not in missing:
            missing.append(dotted)

    # The executed settings are the authoritative record of what the run used. Reporting
    # `materialised: true` while `settings.tasks` is null would let a run pass preflight
    # carrying no statement about which task it collects.
    #
    # `model.adapter` is exempt for the same reason it is exempt above: `adapter: null`
    # means "run the base model", which is a decision the checked-in baseline states out
    # loud. Counting it as a gap would make every base-model run look incomplete.
    if not missing:
        empty = sorted(
            name for name, value in settings.items() if value is None and name not in OPTIONAL_IN_EXECUTED_NAMES
        )
        if empty:
            record["error"] = (
                "this file is materialised but its executed settings are incomplete: "
                + ", ".join(empty)
                + ". A run whose task list, rollout count or model identity is unrecorded "
                "cannot be checked against what it executed."
            )
        record["settings_complete"] = not empty

    record["settings"] = settings
    record["materialised"] = not missing
    if missing:
        record["missing"] = missing
        record["settings_complete"] = False
        record["error"] = (
            "this file is not a materialised run config: it omits "
            + ", ".join(missing)
            + ". The collector resolves no `extends:`, so an overlay-only file cannot drive a run."
        )
    return record


def build_manifest(
    *,
    arm: str,
    run_id: str,
    split: str,
    seed: int | None,
    input_checkpoint: str | None,
    output_checkpoint: str | None,
    note: str | None,
    eval_config: str | None = None,
    # Where this manifest will be written. Used only as a fallback for `artifacts.run_dir`
    # when the executed config names no paths of its own; a manifest describes the
    # directory it lives in, so that directory is always a true answer.
    out_dir: Path | None = None,
    # Whether this manifest authorises a real rollout. `False` (the default) means it is an
    # inspection report: honest about what a run would do, but never accepted by the
    # launcher. `True` means the caller intends to execute, which is what makes a dirty tree
    # and an untracked config file into blockers instead of notes.
    execution_claim: bool = False,
) -> dict[str, Any]:
    if arm not in ARM_CONFIGS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARM_CONFIGS)}")

    protocol = load_protocol()
    pilot = load_pilot()
    now = datetime.now(UTC).isoformat()
    # Read once. `executed_config` hashes the file and parses it, and the `artifacts` block
    # below needs the same parse - reading twice invites the two blocks to disagree.
    executed = executed_config(eval_config) if eval_config else None

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
        # -- the config this run actually executed ------------------------------------
        #    Separate from `effective` on purpose. `effective` describes the training pair
        #    that a GRPO arm would run; a diagnostic smoke runs a materialised eval config,
        #    and Phase 6E's manifest reported 8 rollouts for a run that executed 4 because
        #    nothing recorded the second file. Null when the caller did not name one, which
        #    is honest: this manifest then makes no claim about what ran.
        "executed": executed,
        # -- the rollout-only overlay, recorded so a pilot run names the file that
        #    defined it rather than only the formal protocol it did not execute.
        #    The file hash is unconditional: it answers "did the overlay exist and has
        #    anyone edited it", which is true whether or not this arm used it. The run's
        #    settings are null on other arms, because a baseline manifest asserting
        #    `training_iterations: 0` from a file it never read would be a false claim
        #    dressed up as completeness. -----------------------------------------------
        #
        #    Everything under `values` is read from the overlay *file*, not from the run.
        #    `values.tasks: null` and `values.rollouts_per_task: 8` are that file's shipped
        #    placeholders and are deliberately not reconciled with what executed: the
        #    overlay says what a pilot would use, `executed.settings` says what this run
        #    did. Nothing reads `pilot.values` to decide a run's identity. A run that
        #    executes a materialised eval config never loads this file at all.
        "pilot": {
            "applies": arm in NON_UPDATING_ARMS,
            "values_from": "overlay_file",
            "values_note": (
                "read from the overlay file; placeholders such as tasks: null are "
                "overlay values, not the executed identity - see executed.settings"
            ),
            "file": config_hash(PILOT_CONFIG),
            "name": pilot.get("meta", {}).get("name") if arm in NON_UPDATING_ARMS else None,
            "formal_protocol": (pilot.get("meta", {}).get("formal_protocol") if arm in NON_UPDATING_ARMS else None),
            "performs_parameter_update": (
                (pilot.get("purpose") or {}).get("performs_parameter_update") if arm in NON_UPDATING_ARMS else None
            ),
            "values": {
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
                "concurrency": ((pilot.get("hardware") or {}).get("concurrency") if arm in NON_UPDATING_ARMS else None),
                "heldout_enabled": (
                    (pilot.get("data") or {}).get("heldout_enabled") if arm in NON_UPDATING_ARMS else None
                ),
                "reward_offline_only": (
                    (pilot.get("reward") or {}).get("offline_only") if arm in NON_UPDATING_ARMS else None
                ),
                "reward_v2_computed_inline": (
                    (pilot.get("reward") or {}).get("reward_v2_computed_inline") if arm in NON_UPDATING_ARMS else None
                ),
            },
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
        # -- what this manifest claims. `True` means "a launcher may act on this"; `False`
        #    means "this is a description, not an authorisation". Recorded as its own field
        #    rather than inferred from `executed` being present, because a diagnostic smoke
        #    declares an eval config without ever being allowed to start one.
        "execution_claim": bool(execution_claim),
        # -- what the run intends to write. Populated at manifest creation from the config
        #    the collector will load, so a planned run names its own directory and trace
        #    instead of leaving the reader to infer them from the run id. `log` and
        #    `server_pid` stay null here on purpose: they are facts about a process that has
        #    not started, and a manifest written before a run is the one that matters most.
        "artifacts": _artifacts_block(
            executed,
            out_dir=out_dir if out_dir is not None else REPO_ROOT / "results" / "raw" / run_id,
        ),
    }
    manifest["unresolved"] = unresolved_fields(manifest)
    return manifest


#: Why a field is null, for the three cases that are not gaps. Keys are the dotted paths
#: `unresolved_fields` would otherwise report; values are the reason a reader is owed.
#: Recording the reason is the difference between "nobody decided" and "there is nothing
#: to decide", and the two must not print identically.
NOT_APPLICABLE_REASONS: dict[str, str] = {
    "protocol.grpo.kl_coefficient": (
        "no reference-policy KL term exists in the current controlled implementation "
        "(kl_coefficient_status: NOT_APPLICABLE_CURRENT_IMPLEMENTATION)"
    ),
}

#: Manifest fields that are legitimately null before a run starts. Reported as warnings so
#: a reader knows they are pending rather than missing.
PENDING_AFTER_STARTUP: dict[str, str] = {
    "artifacts.log": "the server writes its log only once it has started",
    "artifacts.server_pid": "the server PID does not exist until the server starts",
}

#: Gaps about the *machine*, not about the executed config. Whether each one blocks depends
#: entirely on what the manifest claims to be. Two manifests can describe the same run from
#: the same dirty tree and one of them must be refused, so the mode is a property of the
#: manifest (`execution_claim`), not of the environment.
#:
#: With `execution_claim: false` the manifest is an inspection report: it exists to be read
#: on a laptop, and a missing GPU or an uncommitted edit is information, not an obstacle.
#: With `execution_claim: true` the manifest authorises a real rollout, and then a dirty
#: tree is disqualifying: the commit it names does not contain the code that would run.
ENVIRONMENTAL_FIELDS: dict[str, tuple[str, str]] = {
    # field: (reason, the reason it matters when a run is actually going to start)
    "git.dirty (tree had uncommitted changes)": (
        "the manifest names a commit that does not contain the code that ran",
        "a run from a dirty tree is not reproducible from the commit it names",
    ),
    "gpu (no GPU detected)": (
        "no GPU on this machine; the manifest records that rather than assuming one",
        "a rollout needs the model server, which needs a GPU",
    ),
}

#: Kept as the flat warning table for CPU inspection, so existing readers keep working.
ENVIRONMENTAL_WARNINGS: dict[str, str] = {field: reason for field, (reason, _) in ENVIRONMENTAL_FIELDS.items()}

#: The two modes a manifest can be written in. `True` means "this authorises a rollout";
#: `False` means "this is a description for a reader". The distinction is recorded in the
#: manifest itself so a launcher can check the claim rather than infer it from which flags
#: the operator happened to pass.
EXECUTION_CLAIMS: frozenset[bool] = frozenset({True, False})


def classify_fields(manifest: dict[str, Any]) -> dict[str, Any]:
    """Split the manifest's gaps into blockers, warnings and not-applicable fields.

    Three different things were previously printed as one `unresolved` list: a value the
    run needs and does not have (a blocker), a value that arrives later by construction (a
    warning), and a value that does not exist in this implementation and never will (not
    applicable). Collapsing them meant a KL coefficient that can never be filled in looked
    as alarming as a missing task list, which is how a real blocker gets scrolled past.

    The fourth distinction is the one that shapes this function: a gap can be blocking *for
    an execution claim* and harmless for an inspection report. `git.dirty` is the clearest
    case. Refusing to write a manifest from a dirty tree made the tool unusable during
    development; accepting a manifest from a dirty tree *as authorisation to run* is what is
    actually unsafe. So the classification reads `execution_claim` and a dirty tree only
    blocks when the manifest claims a run.

    Only `blockers` may stop a run. `warnings` are true, expected states; `not_applicable`
    entries are recorded so a reader does not go looking for a decision that was never
    available to make.
    """
    raw_gaps = unresolved_fields(manifest)
    claims_execution = manifest.get("execution_claim") is True

    not_applicable = [{"field": field, "reason": reason} for field, reason in NOT_APPLICABLE_REASONS.items()]
    na_fields = set(NOT_APPLICABLE_REASONS)

    warnings = [{"field": field, "reason": reason} for field, reason in PENDING_AFTER_STARTUP.items()]
    warned_fields = set(PENDING_AFTER_STARTUP)
    blockers = [field for field in raw_gaps if field not in na_fields and field not in warned_fields]

    # Environmental gaps: a warning for an inspection report, a blocker for an execution
    # claim. `git.dirty` is checked here rather than only in `executable_report` so that one
    # function owns the answer and the launcher cannot reach a different conclusion.
    for field, (inspection_reason, execution_reason) in ENVIRONMENTAL_FIELDS.items():
        if field not in blockers:
            continue
        if claims_execution:
            blockers[blockers.index(field)] = f"{field}: {execution_reason}"
        else:
            blockers.remove(field)
            warnings.append({"field": field, "reason": inspection_reason})
            warned_fields.add(field)

    # An untracked source or config file is reported separately from `dirty`: it cannot
    # change what a commit contains, but it *can* change what a run picks up, so it must not
    # be invisible either. It blocks an execution claim for that reason.
    untracked = [path for path in (manifest.get("git", {}).get("untracked") or []) if path]
    if untracked:
        detail = f"untracked files not under an ignored path: {', '.join(sorted(untracked))}"
        if claims_execution:
            blockers.append(detail + " (they may change what the run picks up)")
        else:
            warnings.append({"field": "git.untracked", "reason": detail})

    return {
        "blockers": blockers,
        "warnings": warnings,
        "not_applicable": not_applicable,
        # Kept for readers of the older one-list shape. It is the *union*, not the blockers,
        # so an existing consumer still sees every gap exactly once.
        "unresolved": raw_gaps,
    }


def executable_report(manifest: dict[str, Any]) -> dict[str, Any]:
    """Whether this manifest describes a run that could be started, and why not if it cannot.

    A manifest that names a config has to name a *runnable* one. `executed.materialised`
    answers "did the file contain the keys", but a materialised file whose task list or
    rollout count parsed as null would still pass that test while leaving the run
    unattributable. This closes the last gap: the executed settings must be present, the
    artifact paths the collector opens must be declared, and the run must be the kind of run
    the arm claims to be.

    Deliberately does not look at `artifacts.log` or `artifacts.server_pid`. Both are null
    before a server starts, by design, and treating their nullness as a failure would make
    every manifest written before a run look broken.

    Two questions are answered here, and the manifest's `execution_claim` decides which one:

    * *May anything act on this?* - blocked by the executed config, the settings, the
      declared outputs, and the shape of the run. These are wrong regardless of mode, so
      they block an inspection report too: an inspection of a broken config is a false
      report.
    * *May a rollout start from this?* - additionally requires a clean tracked tree, no
      untracked source or config files, and a commit that is still HEAD. A dirty tree does
      not make the *description* wrong; it makes the *authorisation* worthless.

    That is why `git.dirty` is only mode-dependent and not simply a blocker: refusing to
    write a manifest on a laptop mid-edit is what made the tool unusable, and accepting one
    as a licence to run is what is actually unsafe. The claim separates the two.

    The returned `blockers` are hard stops. `warnings` and `not_applicable` are reported so
    the caller can print them, but neither can stop a run.
    """
    claims_execution = manifest.get("execution_claim") is True
    executed = manifest.get("executed")
    if executed is None:
        # No `--eval-config` was given, so there is no executed config to vouch for. That is
        # honest for a training arm. But an *execution claim* with nothing to execute is a
        # contradiction: the manifest says a launcher may start a run it cannot describe.
        if claims_execution:
            return {
                "ok": False,
                "reason": "execution_claim is true but no exec config was recorded",
                "blockers": ["execution_claim is true yet executed is null: there is no config to execute"],
                "warnings": [],
                "not_applicable": [],
                "status": "preflight_failed",
            }
        return {
            "ok": True,
            "reason": None,
            "blockers": [],
            "warnings": [],
            "not_applicable": [],
            "status": "no_execution_claim",
        }

    settings = executed.get("settings") or {}
    artifacts = manifest.get("artifacts") or {}
    blockers: list[str] = []

    # -- the config itself -----------------------------------------------------------
    if not executed.get("materialised"):
        blockers.append("executed.materialised is false: the file is an overlay, not a run config")
    if executed.get("settings_complete") is False:
        empty = sorted(name for name, value in settings.items() if value is None)
        blockers.append("executed.settings is incomplete: " + (", ".join(empty) or "unknown field"))

    # -- the settings a run cannot start without -------------------------------------
    if not settings.get("tasks"):
        blockers.append("no executed task: executed.settings.tasks is null or empty")
    if not settings.get("rollouts_per_task"):
        blockers.append("rollouts_per_task is missing from executed.settings")
    if not settings.get("model_name") or not settings.get("model_revision"):
        blockers.append("model identity is incomplete: executed.settings.model_name/model_revision must both be set")

    # -- the paths the collector will open -------------------------------------------
    for mapping in OUTPUT_MAP:
        if mapping.required and artifacts.get(mapping.field) is None:
            blockers.append(f"required output path missing: output.{mapping.key} (to artifacts.{mapping.field})")
    for key in ("run_dir", "trace"):
        if artifacts.get(key) is None:
            blockers.append(f"artifacts.{key} is null")

    # -- the shape of the run --------------------------------------------------------
    if settings.get("heldout_enabled") is not False:
        blockers.append("heldout_enabled must be false: the held-out split is sealed")
    if settings.get("optimizer_enabled"):
        blockers.append("optimizer_enabled is true: a rollout-only diagnostic run updates nothing")
    if settings.get("training_iterations"):
        blockers.append(
            f"training_iterations is {settings['training_iterations']}: "
            "a rollout-only diagnostic run performs no updates"
        )
    if settings.get("expect_output_checkpoint"):
        blockers.append("expect_output_checkpoint is true: a rollout-only run produces no checkpoint")

    # -- an execution claim is refused on evidence of drift ---------------------------
    # These three are cheap to check at creation and catastrophic to get wrong later: a run
    # launched from a commit that no longer exists, a config edited after the manifest was
    # written, or a stray config file that the collector might load instead. A launcher
    # re-checks all of them (scripts/run_in_serving_env.py), because a manifest can be
    # written on Monday and used on Tuesday.
    if claims_execution:
        if executed.get("sha256") is None:
            blockers.append("execution_claim is true but executed.sha256 is null: the config cannot be re-verified")

    # -- anything else still unresolved ----------------------------------------------
    classification = classify_fields(manifest)
    for field in classification["blockers"]:
        # These are recorded gaps the specific checks above may not cover (a missing GPU
        # row, an overlay that vanished, an untracked file). They block because nothing has
        # established that they are safe; a warning would be a claim nobody made.
        blockers.append(f"unresolved: {field}")

    reason = None
    if blockers:
        reason = f"{len(blockers)} blocking condition(s); the manifest does not describe a runnable execute step"
    return {
        "ok": not blockers,
        "reason": reason,
        "blockers": blockers,
        "warnings": classification["warnings"],
        "not_applicable": classification["not_applicable"],
        "status": "executable" if not blockers else "preflight_failed",
    }


def _artifacts_block(executed: dict[str, Any] | None, *, out_dir: Path) -> dict[str, Any]:
    """Where this run writes its evidence, taken from the config it will execute.

    Every field is nullable because every field is a claim: a run with no materialised
    config has no declared paths, and saying so is more useful than guessing
    `results/raw/<run_id>/rollouts.jsonl` from the run id alone. When the executed block
    does declare them, they are recorded verbatim - they are the paths the collector will
    actually open, so the manifest and the run cannot drift apart.

    `run_dir` falls back to the caller's run directory only when the config names none:
    the manifest is written *into* that directory, so it is always the right answer for a
    manifest that exists.
    """
    declared = (executed or {}).get("artifacts") or {}
    block: dict[str, Any] = {
        "run_dir": declared.get("run_dir"),
        "trace": declared.get("trace"),
        "private_trace": declared.get("private_trace"),
        "failures": declared.get("failures"),
        "summary": declared.get("summary"),
        # Facts about a process that has not started. Filled by the runner, not guessed here.
        "log": None,
        "server_pid": None,
    }
    if block["run_dir"] is None:
        block["run_dir"] = str(out_dir.relative_to(REPO_ROOT)) if out_dir.is_relative_to(REPO_ROOT) else str(out_dir)
    return block


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
    parser.add_argument(
        "--eval-config",
        default=None,
        help=(
            "Repo-relative path to the materialised eval config the run will actually execute "
            "(e.g. results/raw/<run>/merged_config.yaml). Recorded, with its hash and settings, "
            "in the `executed` block. Diagnostic smokes do not run the training pair, so without "
            "this the manifest describes a different file from the one the collector loads."
        ),
    )
    parser.add_argument("-o", "--out", default=None, help="Run directory (default: results/raw/<run_id>/)")
    parser.add_argument("--force", action="store_true", help="Supersede an existing, differing manifest")
    parser.add_argument(
        "--execution-claim",
        action="store_true",
        help=(
            "This manifest authorises a real rollout. A dirty tracked tree then becomes a "
            "blocker (exit 2) instead of a warning, because the commit it names would not "
            "contain the code that ran. Without it the manifest is an inspection report: "
            "written for a reader, never accepted by the launcher."
        ),
    )
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
        eval_config=args.eval_config,
        out_dir=out_dir,
        execution_claim=args.execution_claim,
    )

    # Executability is decided *before* anything touches the filesystem.
    #
    # Policy A: validate first, write nothing when the run could not start. Two reasons this
    # beats writing a `preflight_failed` manifest and exiting 2 anyway. (1) A manifest on
    # disk is read later by people and by `--finish`, and one that exists but describes an
    # unstartable run is a trap: the run directory looks prepared. (2) It keeps the invariant
    # that took the Phase 6I preflight down - `outcome.status: planned` must never appear on
    # a run that cannot start - true by construction rather than by remembering to special-case
    # it. Nothing is superseded, nothing needs --force, and a failed preflight leaves the
    # directory exactly as it was.
    executable = executable_report(manifest)
    # Recorded even on success (as `[]`) so the launcher can distinguish "nothing was
    # blocking" from "this manifest predates the field" without guessing.
    manifest["execution_blockers"] = list(executable["blockers"])
    if not executable["ok"]:
        print(f"refusing: {executable['reason']}", file=sys.stderr)
        for blocker in executable["blockers"]:
            print(f"  BLOCKER: {blocker}", file=sys.stderr)
        for warning in executable["warnings"]:
            print(f"  warning: {warning['field']} - {warning['reason']}", file=sys.stderr)
        for item in executable["not_applicable"]:
            print(f"  not applicable: {item['field']} - {item['reason']}", file=sys.stderr)
        print(
            f"refusing: nothing was written to {target}. Fix the config and re-run; "
            "there is no --force that makes an unstartable run startable.",
            file=sys.stderr,
        )
        return 2

    try:
        path, wrote = write_manifest(manifest, out_dir, force=args.force)
    except FileExistsError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1

    print(("wrote " if wrote else "already present ") + str(path))
    # Stated in both directions: a reader must not have to infer the claim from which flags
    # were passed, and `no_execution_claim: true` is the sentence that stops someone
    # treating an inspection report as a licence to start a run.
    if manifest["execution_claim"]:
        print(f"  executable: {str(executable['ok']).lower()} ({executable['status']})")
        print("  execution_claim: true (a launcher may act on this manifest)")
    else:
        print("  executable: false")
        print("  no_execution_claim: true (inspection only; the rollout launcher will refuse this)")
    if executable["blockers"]:
        out = sys.stderr
        for blocker in executable["blockers"]:
            print(f"  BLOCKER: {blocker}", file=out)
    for warning in executable["warnings"]:
        print(f"  warning: {warning['field']} - {warning['reason']}")
    for item in executable["not_applicable"]:
        print(f"  not applicable: {item['field']}")
    untracked = manifest.get("git", {}).get("untracked") or []
    if untracked:
        print(f"  untracked (not ignored, {len(untracked)}): " + ", ".join(sorted(untracked)))
    if not manifest["gpu"]["present"]:
        print("  note: no GPU detected on this machine (normal off the training box)")
    if args.json:
        print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
