#!/usr/bin/env python
"""Materialise an eval config overlay onto its base, and record what the run will use.

Why this exists. `scripts/collect_baseline_rollouts.py` loads exactly ONE file via
`BaselineConfig.load()` and resolves no `extends:` directive at all - a missing key raises
`CollectorError` ("a missing key is a broken config, not a value to default"). The Phase 6
smoke overlays are therefore *not runnable on their own*: they carry only the fields that
differ from `configs/eval/baseline_phase6.yaml`.

Phase 6E materialised its overlay by hand and saved the result as
`results/raw/<run>/phase6e_merged_config.yaml`. That file is the one the collector read, and
it is the file the manifest should fingerprint. Doing the same merge by hand twice invites
the two copies to drift, so it is done here, once, deterministically.

The merge is deliberately narrow:

  * mappings merge recursively;
  * a scalar or list in the overlay replaces the base's value outright;
  * `extends` is consumed, not copied into the output.

That is the same rule `_deep_merge()` in `scripts/run_manifest.py` applies to the training
pair. It is re-stated rather than imported because the two live in different scripts and a
shared helper would couple the eval path to the training-manifest path for no benefit.

    uv run python scripts/materialise_eval_config.py \\
        --overlay configs/eval/phase6g_diagnostic_smoke.yaml \\
        --out results/raw/<run_id>/merged_config.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "configs/eval/baseline_phase6.yaml"


class MaterialiseError(RuntimeError):
    """A config could not be materialised, or would not be runnable if it were."""


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge `overlay` onto a deep copy of `base`, recursing into mappings only."""
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MaterialiseError(f"no such config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def materialise(overlay_path: Path, base_path: Path | None = None) -> dict[str, Any]:
    """The overlay merged onto its base. `extends` selects the base when not given."""
    overlay = _load(overlay_path)
    declared = overlay.get("extends")
    if base_path is None:
        if not declared:
            raise MaterialiseError(
                f"{overlay_path} has no `extends:` and no --base was given; there is nothing to merge onto"
            )
        base_path = REPO_ROOT / str(declared)
    elif declared and (REPO_ROOT / str(declared)).resolve() != Path(base_path).resolve():
        raise MaterialiseError(
            f"{overlay_path} declares `extends: {declared}` but --base is {base_path}; refusing to "
            "silently merge onto a different base than the file names"
        )
    return _deep_merge(_load(base_path), overlay)


def _require(raw: dict[str, Any], dotted: str) -> Any:
    node: Any = raw
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise MaterialiseError(f"materialised config is missing '{dotted}'; it is not runnable")
        node = node[part]
    return node


def validate(raw: dict[str, Any]) -> dict[str, Any]:
    """The settings a reader needs, or an error naming the first that is absent.

    Every key here is one `BaselineConfig` reads through `_get()`, which raises rather than
    defaulting. Checking them now means a broken merge fails at materialise time, next to
    the file that caused it, instead of minutes later inside the collector.
    """
    required = (
        "model.name",
        "model.revision",
        "sampling.temperature",
        "sampling.top_p",
        "sampling.seed",
        "run.task_subset",
        "run.rollouts_per_task",
        "output.run_id",
        "output.rollouts",
        "output.private",
        "context.max_model_len",
        "context.tokenizer_path",
        "tasks.split",
        "reward.live",
        "server.base_url",
    )
    for dotted in required:
        _require(raw, dotted)

    split = _require(raw, "tasks.split")
    if split != "train":
        raise MaterialiseError(
            f"tasks.split is {split!r}; Phase 6 runs on train only. The held-out split is "
            "sealed (docs/heldout_seal.md)."
        )
    if _require(raw, "model.adapter") is not None:
        raise MaterialiseError("model.adapter is set; a diagnostic smoke must run the base model")

    return {
        "run_id": _require(raw, "output.run_id"),
        "tasks": _require(raw, "run.task_subset"),
        "rollouts_per_task": _require(raw, "run.rollouts_per_task"),
        "diagnostic_only": _require(raw, "run.diagnostic_only"),
        "model_name": _require(raw, "model.name"),
        "model_revision": _require(raw, "model.revision"),
        "temperature": _require(raw, "sampling.temperature"),
        "top_p": _require(raw, "sampling.top_p"),
        "seed": _require(raw, "sampling.seed"),
        "split": split,
        "reward_live": _require(raw, "reward.live"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overlay", required=True, help="Repo-relative overlay config to materialise")
    parser.add_argument(
        "--base", default=None, help=f"Base config (default: the overlay's `extends:`, else {DEFAULT_BASE})"
    )
    parser.add_argument("--out", required=True, help="Repo-relative path to write the materialised config")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args(argv)

    overlay = REPO_ROOT / args.overlay
    base = REPO_ROOT / args.base if args.base else None
    out = REPO_ROOT / args.out

    try:
        merged = materialise(overlay, base)
        settings = validate(merged)
    except MaterialiseError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2

    # The output is never a silent overwrite: a materialised config is evidence of what a
    # specific run executed, so clobbering one would rewrite history.
    if out.exists() and not args.force:
        print(f"refusing: {out} already exists (pass --force to overwrite)", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(merged, handle, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"wrote {out}")
    print(f"  base      : {base or REPO_ROOT / str(_load(overlay).get('extends'))}")
    print(f"  overlay   : {overlay}")
    for key in ("run_id", "tasks", "rollouts_per_task", "diagnostic_only", "model_name", "seed"):
        print(f"  {key:12}: {settings[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
