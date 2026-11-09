#!/usr/bin/env python
"""Write the Phase 7 Part C training-input manifest.

A result is only interpretable if the thing that produced it can be named. Phase
7 consumes four rollouts that already exist, so the manifest's job is narrower
than `run_manifest.py`'s: it fingerprints *the exact rows that will be adapted*
and records the identities that must not drift between writing the manifest and
starting the GPU run.

    uv run python scripts/build_phase7_input_manifest.py \\
        --rollouts results/raw/phase6m_.../rollouts.jsonl \\
        --output results/phase7/input_manifest.json

Three properties worth stating, because they are why this is code:

**The rows are hashed, not described.** Each selected row's SHA-256 is recorded
alongside the file's. A manifest that merely named the run id would still be
correct after somebody edited a reward in place.

**Absence is recorded as absence.** The Phase 6M analysis JSON lives under
`results/analysis/`, which is gitignored and was never committed. It is absent on
this machine. The manifest says so explicitly rather than omitting the field,
because omission is indistinguishable from "not checked". It is not reconstructed:
a rebuilt file would not be the run's original analysis, and presenting it as
such would be fabrication.

**Nothing here is a measurement.** No loss, no gradient, no VRAM. Those go in
`training_metrics.json` after the step. A manifest that carried results would
invite editing it to match a number that turned out inconvenient.

Exit codes: 0 written / already-present-and-matching, 1 refused overwrite, 2 bad input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from analysis.versions import repo_commit
from training.grpo_math import DegenerateGroupError, group_advantages
from training.phase7_guards import (
    EXPECTED_OPTIMIZER_STEPS,
    PERMITTED_ITERATIONS,
    PERMITTED_MODEL_NAME,
    PERMITTED_MODEL_REVISION,
    PERMITTED_REWARDS,
    REQUIRED_ROLLOUT_COUNT,
    Phase7Violation,
    normalise_for_guard,
    seed_of,
    validate_rollout_group,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The Phase 6M run this manifest may describe. Recorded so a manifest cannot be
#: quietly pointed at a different group.
SOURCE_RUN_ID = "phase6m_bugfix_offbyone_pagination_3b_20261109T104748Z"

#: Identities read from that run's `merged_config.yaml`. Copied, not re-derived:
#: re-deriving them from the current configs would record today's settings and
#: mislabel them as the ones the rollouts were produced under.
SOURCE_PROMPT_VERSION = "baseline_phase6_v1"
SOURCE_TOOLS_VERSION = "coding_env_tools_v3"
SOURCE_REWARD_LIVE = "reward_v1"
SOURCE_REWARD_ANALYSED = "reward_v2"
SOURCE_SCHEMA_VERSION = 3
SOURCE_ROLLOUT_REPO_COMMIT = "14eaf8f4420d0818f3eded599cc909710c9beb92"
SOURCE_VERIFIER_ENTRYPOINT = "envs.coding_env.verifier.verify_sandbox"

#: The analysis file the collector would have written, and which is not on disk.
ABSENT_ANALYSIS_PATH = f"results/analysis/{SOURCE_RUN_ID}.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_row_sha256(row: dict[str, Any]) -> str:
    """Stable digest of one row: sorted keys, no whitespace, UTF-8."""
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(rollouts_path: Path, *, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    raw_lines = [line for line in rollouts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in raw_lines]

    # Reuse the runtime guard so the manifest and the trainer cannot disagree about
    # what a permitted group is. The guard works on flat fields; the trace nests
    # them, so project first and validate the projection.
    flat = normalise_for_guard(rows)

    try:
        validate_rollout_group(flat)
    except Phase7Violation as error:
        raise SystemExit(f"refusing to write a manifest for a non-Phase-7 group: {error}") from error

    # Order by the recorded `rollout_index`, not by file position. They agree for
    # Phase 6M, and if they ever do not, the recorded index is what defines the
    # group.
    ordered = sorted(zip(flat, rows, strict=True), key=lambda pair: pair[0]["rollout_index"])

    rewards = [float(flat_row["reward"]) for flat_row, _ in ordered]

    try:
        advantages, reward_mean, reward_std = group_advantages(rewards)
    except DegenerateGroupError as error:  # pragma: no cover - the guard above rejects this first
        raise SystemExit(f"degenerate group: {error}") from error

    analysis_path = REPO_ROOT / ABSENT_ANALYSIS_PATH

    return {
        "manifest_kind": "phase7_training_input",
        "manifest_version": "1",
        "generated_utc": datetime.now(UTC).isoformat(),
        "generated_by": "scripts/build_phase7_input_manifest.py",
        "newly_generated": True,
        "source_run": {
            "run_id": SOURCE_RUN_ID,
            "rollouts_path": str(rollouts_path.relative_to(REPO_ROOT))
            if rollouts_path.is_absolute() and rollouts_path.is_relative_to(REPO_ROOT)
            else str(rollouts_path),
            "rollouts_sha256": file_sha256(rollouts_path),
            "rollouts_line_count": len(raw_lines),
            "rollouts_byte_count": rollouts_path.stat().st_size,
            "rollout_repo_commit": SOURCE_ROLLOUT_REPO_COMMIT,
            "schema_version": SOURCE_SCHEMA_VERSION,
            "prompt_version": SOURCE_PROMPT_VERSION,
            "tools_version": SOURCE_TOOLS_VERSION,
            "verifier_entrypoint": SOURCE_VERIFIER_ENTRYPOINT,
            "reward_live": SOURCE_REWARD_LIVE,
            "reward_analysed": SOURCE_REWARD_ANALYSED,
            "heldout": False,
            "task_split": "train",
        },
        "absent_source_artifacts": [
            {
                "path": ABSENT_ANALYSIS_PATH,
                "present": analysis_path.exists(),
                "note": (
                    "Absent on this machine. `results/` is gitignored, so this file was never "
                    "committed. It is recorded as absent and is deliberately NOT reconstructed: a "
                    "rebuilt analysis would not be the original run's output."
                ),
            }
        ],
        "group": {
            "task_id": ordered[0][0]["task_id"],
            "rollout_count": len(rows),
            "required_rollout_count": REQUIRED_ROLLOUT_COUNT,
            "canonical_order": "rollout_index ascending",
            "rewards": rewards,
            "expected_rewards": list(PERMITTED_REWARDS),
            "rewards_match_canonical_sequence": rewards == list(PERMITTED_REWARDS),
            "reward_mean": reward_mean,
            "reward_std_population": reward_std,
            "advantages": advantages,
            "distinct_reward_values": len(set(rewards)),
            "rows": [
                {
                    "position": position,
                    "rollout_index": flat_row["rollout_index"],
                    "rollout_id": raw_row.get("rollout_id"),
                    "sampling_seed": seed_of(raw_row),
                    "reward": float(flat_row["reward"]),
                    "submitted": (raw_row.get("result") or {}).get("submitted"),
                    "termination_reason": raw_row.get("termination_reason"),
                    "error": raw_row.get("error"),
                    "row_sha256": canonical_row_sha256(raw_row),
                }
                for position, (flat_row, raw_row) in enumerate(ordered)
            ],
        },
        "model": {
            "name": PERMITTED_MODEL_NAME,
            "revision": PERMITTED_MODEL_REVISION,
            "local_files_only": True,
            "seven_b_prohibited": True,
            "one_point_five_b_prohibited": True,
        },
        "execution": {
            "iterations": PERMITTED_ITERATIONS,
            "expected_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
            "lora_only": True,
            "existing_adapter_prohibited": True,
            "rollout_generation_disabled": True,
            "evaluation_generation_disabled": True,
            "heldout_evaluation_disabled": True,
            "multi_iteration_prohibited": True,
            "kl_status": "NOT_APPLICABLE_CURRENT_IMPLEMENTATION",
        },
        "repository": {
            "commit_at_manifest_creation": repo_commit(),
        },
        "host": hardware,
        "not_established": [
            "No Phase 7 metric is recorded here; a manifest that carried results would invite editing.",
            "The 6M analysis JSON is absent and was not reconstructed.",
            "Four rollouts and one optimizer step cannot establish a trend.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing manifest.")
    args = parser.parse_args(argv)

    if not args.rollouts.is_file():
        print(f"error: rollouts file not found: {args.rollouts}", file=sys.stderr)
        return 2

    manifest = build_manifest(args.rollouts.resolve())

    if args.output.exists() and not args.force:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("source_run", {}).get("rollouts_sha256") == manifest["source_run"]["rollouts_sha256"]:
            print(f"manifest already present and matching: {args.output}")
            return 0
        print(
            f"error: {args.output} exists and describes a different rollout file. "
            "Refusing to overwrite; pass --force if this is deliberate.",
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.output}")
    print("rewards:", manifest["group"]["rewards"])
    print("advantages:", manifest["group"]["advantages"])
    print("rollouts sha256:", manifest["source_run"]["rollouts_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
