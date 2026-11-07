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
import subprocess
from pathlib import Path
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
