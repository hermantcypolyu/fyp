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


def _run_dir_for(out: Path) -> str:
    """The repo-relative run directory a materialised config at `out` belongs to.

    Returns POSIX form because the paths are compared as POSIX paths elsewhere (the identity
    check in `execution_gate`), and a Windows-style separator here would make every path
    look like it lived outside the directory it was written into.
    """
    try:
        return out.parent.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise MaterialiseError(
            f"{out} is not inside the repository, so its run directory has no relative name"
        ) from exc


def _require(raw: dict[str, Any], dotted: str) -> Any:
    node: Any = raw
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise MaterialiseError(f"materialised config is missing '{dotted}'; it is not runnable")
        node = node[part]
    return node


def apply_run_identity(raw: dict[str, Any], run_id: str, run_dir: str) -> dict[str, Any]:
    """Point every run-identity field at one run id and one directory, in place.

    Why this is explicit rather than inferred from `--out`. The overlays live in `configs/`
    and are committed, so their `output:` block names whichever run last exercised them. A
    materialised copy inherits those values verbatim, and Phase 6I hit exactly that: the
    config for a fresh run still declared the *previous* attempt's `output.run_id` and all
    three of its output paths. Merging is not enough - the identity has to be replaced.

    Every path is rewritten, not just `run_id`, because the collector opens the paths, not
    the id. Overriding the id alone would produce a config that looks like it belongs to the
    new run while writing its rollouts into the old run's directory, which is worse than the
    mismatch it was meant to fix: the evidence would silently merge two attempts.

    The four outputs are fixed here, by name, rather than discovered. A general rewrite of
    "any path under the old run directory" would silently miss a path that pointed somewhere
    else entirely, which is a case worth failing on rather than repairing quietly.
    """
    if not isinstance(run_id, str) or not run_id.strip() or "/" in run_id:
        raise MaterialiseError(f"run id {run_id!r} is not a usable directory name")
    run_id = run_id.strip()
    run_dir = run_dir.rstrip("/")

    output = raw.get("output")
    if not isinstance(output, dict):
        raise MaterialiseError("materialised config has no `output:` mapping to bind to this run")

    output["run_id"] = run_id
    output["rollouts"] = f"{run_dir}/rollouts.jsonl"
    output["failures"] = f"{run_dir}/rollouts_failures.jsonl"
    output["private"] = f"{run_dir}/rollouts_private.jsonl"
    # Analysis output is not under `results/raw/`, so it is derived from the run directory's
    # parent rather than assumed to be a sibling of the rollouts file.
    output["analysis"] = f"results/analysis/{run_id}.json"
    return raw


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
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run id to bind into output.run_id and every output path (defaults to no run identity: refused)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    args = parser.parse_args(argv)

    overlay = REPO_ROOT / args.overlay
    base = REPO_ROOT / args.base if args.base else None
    out = REPO_ROOT / args.out

    try:
        merged = materialise(overlay, base)
        # Bind the identity before validating, so validation reports the values a run would
        # actually use rather than the overlay's inherited ones.
        run_dir = _run_dir_for(out)
        apply_run_identity(merged, args.run_id, run_dir)
        settings = validate(merged)
    except MaterialiseError as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2

    # The output path has to agree with the run it now claims to be, or the file would be
    # written into a directory that its own identity does not name.
    if out.parent.name != args.run_id and out.parent.parent.name != args.run_id:
        print(
            f"refusing: --out {args.out} is not inside a directory named {args.run_id!r}; "
            "the materialised config would name a run it is not stored under",
            file=sys.stderr,
        )
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
    # The paths are printed because they are the thing that went wrong in Phase 6I: the file
    # looked right and only the identity fields disagreed with the directory it sat in.
    bound = merged.get("output") or {}
    for key in ("rollouts", "private", "failures", "analysis"):
        print(f"  output.{key:9}: {bound.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
