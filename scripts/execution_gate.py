"""The gate a manifest must pass before any rollout is allowed to start.

Why this is separate from `run_manifest.py`
-------------------------------------------
`run_manifest.py` decides whether a manifest is *internally* coherent: does the config it
names exist, are the settings present, do the output paths resolve. That is a question about
a file, and it can be answered on a laptop with no GPU.

This module asks the different question a launcher has to answer: *is the world still the
one this manifest describes?* A manifest is written before a run and consumed later, and in
between someone can commit, edit the config, or leave a stray file behind. Checking that at
creation time is not enough, because the whole risk is that the interval is non-zero.

The two questions are deliberately not merged. Merging them would mean either the creation
step needs to know the current HEAD (it does, but only at creation) or the launcher needs to
re-derive every internal invariant (it should not - it should trust the manifest it was
handed, or refuse it). `verify_execution_claim()` therefore re-checks a small, cheap,
independently-verifiable set: the claim, the recorded commit, the config hash, and the
presence of the task and rollout counts.

Exit codes
----------
`run_in_serving_env.py` reserves 0 for success and 3/4 for environment problems. A manifest
that does not authorise a run is 5, so a caller can tell "the environment is broken" from
"you did not have permission to run" without parsing stderr.

CPU-only: these checks touch only git, the filesystem and the manifest dict.
"""

from __future__ import annotations

import hashlib
import posixpath
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Exit code for "the manifest does not authorise this run". Distinct from the serving
#: environment's 3 and 4 because the two failures have different remedies.
EXIT_NOT_AUTHORISED = 5


def _git(*args: str, cwd: Path) -> str | None:
    """Run a git command, returning stdout or None when git is unusable."""
    try:
        proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def current_head(cwd: Path | None = None) -> str | None:
    return _git("rev-parse", "HEAD", cwd=cwd or REPO_ROOT)


def tracked_tree_changes(cwd: Path | None = None) -> list[str] | None:
    """Tracked files that differ from HEAD, or None when git is unusable.

    `--untracked-files=no` is the point: a rollout writes into `results/`, which is ignored,
    and an untracked run artefact must not look like an uncommitted source edit.
    """
    listing = _git("status", "--porcelain", "--untracked-files=no", cwd=cwd or REPO_ROOT)
    if listing is None:
        return None
    return [line for line in listing.splitlines() if line.strip()]


def untracked_paths(cwd: Path | None = None) -> list[str] | None:
    """Untracked files git does not ignore, or None when git is unusable."""
    listing = _git("ls-files", "--others", "--exclude-standard", cwd=cwd or REPO_ROOT)
    if listing is None:
        return None
    return [line for line in listing.splitlines() if line.strip()]


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


#: Every `output.*` path a run writes, and the manifest artifact field it is recorded under.
#: The run-directory identity is checked over all of them: a rollout whose public trace lands
#: in one run's directory and whose private trace lands in another is not one run, and the
#: evidence cannot be assembled afterwards.
#:
#: `output.analysis` is deliberately absent. It is real - the collector writes a per-run
#: analysis file - but it is not a manifest artifact field (`OUTPUT_MAP` in
#: `scripts/run_manifest.py` maps `output.rollouts` to both `trace` and `run_dir`, and derives
#: `summary` as a *sibling* of the trace), and it legitimately lives in `results/analysis/`
#: rather than inside the run directory. Requiring it under the run directory would fail every
#: correctly-prepared run. It is still bound to the run id by its filename, which the
#: materialiser enforces; it is simply not evidence that has to sit in the run directory.
OUTPUT_PATH_FIELDS: tuple[tuple[str, str], ...] = (
    ("output.rollouts", "trace"),
    ("output.private", "private_trace"),
    ("output.failures", "failures"),
)


def _posix(value: Any) -> str | None:
    """A repo-relative POSIX path with `.` and `..` resolved, or None if unusable.

    `normpath` rather than `PurePosixPath` alone: `PurePosixPath("a/b/../c")` reports its
    parents as `a/b/..` and `a/b`, so a path that escapes the run directory with `..` would
    be judged to be *under* it. The path being compared is a claim about where evidence
    lands, so it has to be normalised before it is compared, not after.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    return posixpath.normpath(value.strip())


def run_identity_checks(
    *,
    cli_run_id: str | None,
    manifest_run_id: Any,
    executed_run_id: Any,
    config_run_id: Any,
    run_dir: Any,
    output_paths: dict[str, Any],
) -> list[str]:
    """Every way the run's identity can fail to name one directory.

    The contract is one sentence: the CLI run id, the manifest's `run_id`, the executed
    config's `output.run_id`, and the basename of the result directory are the same string,
    and every output path the run writes is inside that directory.

    It exists because a materialised config inherits `output.run_id` from a committed overlay.
    Phase 6I prepared a run under a fresh id while its config still named the previous
    attempt, so a rollout started from it would have written rollouts, private rows and
    failure records into the *failed* run's directory - mixing two attempts into one set of
    evidence, and destroying the failed attempt's evidence in the process. Nothing in the
    manifest was individually false; the mismatch was only visible by comparing fields that
    had never been compared.

    Returns a list of reasons; empty means the identity is consistent. Paths are compared as
    POSIX paths with `..` resolved, so `results/raw/x/../y` cannot pass as `results/raw/x`.
    """
    reasons: list[str] = []

    normalised = {
        "manifest run_id": manifest_run_id,
        "executed run_id": executed_run_id,
        "config output.run_id": config_run_id,
    }
    for label, value in normalised.items():
        if not isinstance(value, str) or not value.strip():
            reasons.append(f"run identity: {label} is missing or empty")
        elif cli_run_id is not None and value.strip() != cli_run_id.strip():
            reasons.append(f"run identity: {label} is {value.strip()!r} but the run id is {cli_run_id.strip()!r}")

    declared_dir = _posix(run_dir)
    if declared_dir is None:
        reasons.append("run identity: the result directory is not recorded")
        return reasons

    if cli_run_id is not None:
        basename = PurePosixPath(declared_dir).name
        if basename != cli_run_id.strip():
            reasons.append(
                f"run identity: the result directory basename is {basename!r} but the run id is {cli_run_id.strip()!r}"
            )

    # Every output the run writes has to live under that one directory.
    root = PurePosixPath(declared_dir)
    for key, field in OUTPUT_PATH_FIELDS:
        raw = output_paths.get(key)
        if raw is None:
            # Absence is the caller's business (`run_manifest` blocks on a missing required
            # path itself); a null here is not an identity mismatch.
            continue
        candidate = _posix(raw)
        if candidate is None:
            reasons.append(f"{key} is not a usable path: {raw!r}")
            continue
        if PurePosixPath(candidate) == root or root in PurePosixPath(candidate).parents:
            continue
        reasons.append(
            f"{key} is outside the run directory: {candidate} is not under {root} (recorded as artifacts.{field})"
        )

    return reasons


def verify_run_identity(manifest: dict[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    """Re-read the executed config and check the run identity against the manifest.

    This is the launcher's second opinion. `run_manifest.py` compares the fields when it
    writes the manifest, but a manifest can be written on Monday and used on Tuesday, and
    the specific hazard here is a config *replaced* after the manifest named it. Re-reading
    the file is what makes that detectable: the recorded identity is the manifest's, the
    observed identity is whatever the config says now.

    Returns ``{"ok": bool, "reasons": [...], "checked": {...}}`` like
    `verify_execution_claim`, so the launcher can report both in the same shape.
    """
    import yaml

    root = repo_root or REPO_ROOT
    executed = manifest.get("executed") or {}
    artifacts = manifest.get("artifacts") or {}
    config_rel = executed.get("path")
    checked: dict[str, Any] = {"config_path": config_rel}

    if not config_rel:
        return {
            "ok": False,
            "reasons": ["run identity cannot be checked: no executed config is recorded"],
            "checked": checked,
        }

    path = root / config_rel
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {
            "ok": False,
            "reasons": [f"run identity cannot be checked: the executed config could not be read: {exc}"],
            "checked": checked,
        }

    output = raw.get("output") if isinstance(raw, dict) else None
    output = output if isinstance(output, dict) else {}
    checked["config_run_id"] = output.get("run_id")

    reasons = run_identity_checks(
        cli_run_id=manifest.get("run_id"),
        manifest_run_id=manifest.get("run_id"),
        executed_run_id=(executed.get("settings") or {}).get("run_id"),
        config_run_id=output.get("run_id"),
        run_dir=artifacts.get("run_dir"),
        output_paths={
            "output.rollouts": output.get("rollouts"),
            "output.private": output.get("private"),
            "output.failures": output.get("failures"),
        },
    )
    return {"ok": not reasons, "reasons": reasons, "checked": checked}


def verify_execution_claim(
    manifest: dict[str, Any],
    *,
    repo_root: Path | None = None,
    require_gpu: bool = True,
) -> dict[str, Any]:
    """Decide whether a rollout may start from this manifest.

    Returns ``{"ok": bool, "reasons": [...], "checked": {...}}``. Every failure is collected
    rather than short-circuited, because a launcher operator who fixes one thing at a time
    and re-runs is a launcher operator having a bad afternoon.

    `require_gpu` is a parameter rather than a hard-coded check so the CPU tests can exercise
    the GPU-refusal path on a machine that has no GPU to refuse, and so a future CPU-only
    rollout driver is not silently blocked.
    """
    root = repo_root or REPO_ROOT
    reasons: list[str] = []
    checked: dict[str, Any] = {}

    # -- 1. the manifest has to claim execution at all -------------------------------
    claim = manifest.get("execution_claim")
    checked["execution_claim"] = claim
    if claim is not True:
        reasons.append(
            "execution_claim is not true: this manifest is an inspection report, not an authorisation to start a run"
        )
        # Nothing below can be meaningful for a non-claiming manifest, and reporting a
        # commit mismatch on a document that never claimed to authorise anything would be
        # noise on top of the real reason.
        return {"ok": False, "reasons": reasons, "checked": checked}

    # -- 2. the manifest has to say it is executable ---------------------------------
    outcome = (manifest.get("outcome") or {}).get("status")
    checked["outcome_status"] = outcome
    if outcome == "preflight_failed":
        reasons.append("outcome.status is preflight_failed: the manifest records that it cannot start")

    # -- 3. no recorded blockers -----------------------------------------------------
    # `execution_blockers` is written by the preflight when it refuses; a manifest carrying
    # any is a refusal that someone kept.
    recorded = manifest.get("execution_blockers") or []
    checked["recorded_blockers"] = len(recorded)
    if recorded:
        reasons.append(f"the manifest records {len(recorded)} blocker(s): " + "; ".join(recorded))

    # -- 4. the tracked tree must be clean -------------------------------------------
    changes = tracked_tree_changes(root)
    checked["tracked_changes"] = changes
    if changes is None:
        reasons.append("cannot read the git state: a rollout cannot be attributed to a commit")
    elif changes:
        reasons.append(
            "the tracked tree has uncommitted changes, so the run would not be reproducible "
            f"from the commit the manifest names: {', '.join(changes)}"
        )

    # -- 5. untracked non-ignored files must not be present --------------------------
    stray = untracked_paths(root)
    checked["untracked"] = stray
    if stray:
        reasons.append("untracked files that are not ignored may change what the run picks up: " + ", ".join(stray))

    # -- 6. the commit must still be HEAD --------------------------------------------
    recorded_commit = (manifest.get("git") or {}).get("commit")
    head = current_head(root)
    checked["manifest_commit"] = recorded_commit
    checked["head"] = head
    if recorded_commit is None:
        reasons.append("the manifest records no commit")
    elif head is None:
        reasons.append("cannot resolve HEAD to compare against the manifest's commit")
    elif recorded_commit != head:
        reasons.append(f"the manifest names commit {recorded_commit[:12]} but HEAD is {head[:12]}")

    # -- 7. the config on disk must still match the hash ------------------------------
    executed = manifest.get("executed") or {}
    config_rel = executed.get("path")
    recorded_hash = executed.get("sha256")
    checked["config_path"] = config_rel
    checked["config_sha256"] = recorded_hash
    if not config_rel or not recorded_hash:
        reasons.append("the manifest records no executed config path or hash to re-verify")
    else:
        actual = file_sha256(root / config_rel)
        if actual is None:
            reasons.append(f"the executed config is gone: {config_rel}")
        elif actual != recorded_hash:
            reasons.append(
                f"the executed config changed since the manifest was written: "
                f"{config_rel} is {actual[:12]}, the manifest recorded {recorded_hash[:12]}"
            )

    # -- 8. the run has something to execute -----------------------------------------
    settings = executed.get("settings") or {}
    tasks = settings.get("tasks")
    rollouts = settings.get("rollouts_per_task")
    checked["tasks"] = tasks
    checked["rollouts_per_task"] = rollouts
    if not tasks:
        reasons.append("executed.settings.tasks is null or empty: there is no task to run")
    if not rollouts:
        reasons.append("executed.settings.rollouts_per_task is missing")

    # -- 9. a GPU has to exist, checked immediately before starting -------------------
    if require_gpu:
        gpu = manifest.get("gpu") or {}
        checked["gpu_present"] = gpu.get("present")
        if not gpu.get("present"):
            reasons.append(
                "no GPU was recorded when the manifest was written; a rollout needs the model "
                "server, so check the device again before starting"
            )

    return {"ok": not reasons, "reasons": reasons, "checked": checked}


def format_refusal(result: dict[str, Any]) -> str:
    """A refusal a human can act on: every reason, then the one-line summary."""
    lines = ["refusing to start: this manifest does not authorise a rollout."]
    lines += [f"  - {reason}" for reason in result["reasons"]]
    lines.append("Nothing was started. Generate a manifest with --execution-claim on a clean tree to authorise a run.")
    return "\n".join(lines)
