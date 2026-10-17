"""Shared SWE-Gym candidate verification entry point.

This module reuses the exact verifier implementation used by the CLI so
offline evaluation and RL reward cannot drift into separate scoring paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.verify_swe_gym_pilot import (
    DEFAULT_MANIFEST,
    load_manifest,
    select_instance,
    verify_instance,
)


def verify_candidate(
    *,
    instance_id: str,
    candidate_patch_path: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Verify one agent-generated candidate patch in a clean container."""

    candidate_patch_path = Path(
        candidate_patch_path
    ).resolve()

    if not candidate_patch_path.is_file():
        raise FileNotFoundError(
            candidate_patch_path
        )

    rows = load_manifest(
        Path(manifest_path)
    )

    row = select_instance(
        rows,
        instance_id,
    )

    return verify_instance(
        row=row,
        mode="candidate",
        pull=False,
        candidate_patch_path=candidate_patch_path,
    )
