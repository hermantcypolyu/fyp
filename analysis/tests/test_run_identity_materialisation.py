"""Materialisation binds a config to the one run directory it is written into.

`analysis/tests/test_run_identity.py` pins the *contract* - what a consistent identity means
and who refuses an inconsistent one. This file pins the *producer* side of the same contract:
the functions that stamp a run id and its four output paths onto a freshly merged config.

The distinction matters because the two can fail independently. The gate can be perfectly
correct and still be handed a config that names the wrong directory, which is exactly what
happened in Phase 6I: the committed overlay in `configs/` carried the previous attempt's
`output.run_id` and all three of its evidence paths, `_deep_merge` copied them verbatim, and
the prepared run wrote into `..._20261106T161132Z/` while claiming to be `..._20261107T045749Z/`.

So every case here asks the same question of the producer: after materialisation, does this
config point at the run it claims, and nowhere else? Each line of the answer is held down:

  01  a stale inherited `output.run_id` is replaced with the supplied run id
  02  `output.rollouts` is rewritten under the selected run directory
  03  `output.private` is rewritten under the selected run directory
  04  `output.failures` is rewritten under the selected run directory
  05  the output directory's basename must equal the supplied run id
  06  a mismatched run id and `--out` directory is rejected
  07  an `--out` outside the repository is rejected
  08  a path containing `..` cannot escape the selected run directory
  09  a correctly materialised config passes `run_identity_checks()`
  10  the materialised config contains no stale run identity at all
  11  a failed materialisation leaves no partial output file behind
  12  `output.analysis` may sit outside the raw run directory
  13  the failed-preflight evidence on disk is still byte-identical

CPU-only: the materialiser reads YAML and strings and writes one file. No GPU, vLLM, model
weights, network, NeMo or Ray, and no subprocess.

Temporary directories only. Cases 01-12 build their overlay and base inside `tmp_path`, so a
case that writes a config cannot reach this repository's `results/` or `configs/`. Case 13
reads the committed evidence and writes nothing at all.
"""

from __future__ import annotations

import hashlib
import posixpath
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts import materialise_eval_config
from scripts.execution_gate import run_identity_checks
from scripts.materialise_eval_config import MaterialiseError, _run_dir_for, apply_run_identity, materialise

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The run the materialised config is being prepared for.
NEW_RUN_ID = "phase6i_bugfix_offbyone_pagination_20261108T090000Z"
#: The identity the committed overlay still carries: the *previous* attempt.
STALE_RUN_ID = "phase6i_bugfix_offbyone_pagination_20261106T161132Z"

NEW_RUN_DIR = f"results/raw/{NEW_RUN_ID}"
STALE_RUN_DIR = f"results/raw/{STALE_RUN_ID}"

#: An overlay that reproduces the Phase 6I defect: its `output:` block names another run.
#: Everything else is the minimum `validate()` accepts, so a failure is about identity only.
OVERLAY = f"""
extends: configs/base.yaml
model:
  name: Qwen/Qwen2.5-Coder-1.5B-Instruct
run:
  task_subset: [bugfix_offbyone_pagination]
  rollouts_per_task: 4
  diagnostic_only: true
output:
  run_id: {STALE_RUN_ID}
  rollouts: {STALE_RUN_DIR}/rollouts.jsonl
  failures: {STALE_RUN_DIR}/rollouts_failures.jsonl
  private: {STALE_RUN_DIR}/rollouts_private.jsonl
  analysis: results/analysis/{STALE_RUN_ID}.json
"""

#: The base the overlay extends. Carries the keys the overlay does not set.
BASE = """
model:
  name: placeholder
  revision: 0e3f9a2b1c
  adapter: null
sampling:
  temperature: 0.0
  top_p: 1.0
  seed: 42
run:
  task_subset: [bugfix_offbyone_pagination]
  rollouts_per_task: 4
  diagnostic_only: true
context:
  max_model_len: 8192
  tokenizer_path: tokenizer
tasks:
  split: train
reward:
  live: true
server:
  base_url: http://127.0.0.1:8000/v1
"""


def _write_fixture(root: Path) -> tuple[Path, Path]:
    """Write `base` and `overlay` under `root/configs`, returning their paths."""
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    base = configs / "base.yaml"
    overlay = configs / "overlay.yaml"
    base.write_text(BASE, encoding="utf-8")
    overlay.write_text(OVERLAY, encoding="utf-8")
    return base, overlay


@pytest.fixture
def materialise_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway repository the materialiser believes it is running inside.

    `_run_dir_for` and `main()` both measure paths against the module-level `REPO_ROOT`, and
    refusing an `--out` outside the repository is one of the behaviours under test. Rebinding
    `REPO_ROOT` to a temporary directory is what lets cases 06 and 07 be exercised without
    writing anything into this working tree - the alternative, passing real repo paths, would
    mean a failing case could leave a stray config in `results/`.
    """
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(materialise_eval_config, "REPO_ROOT", root)
    _write_fixture(root)
    return root


def _materialised(repo: Path, run_id: str = NEW_RUN_ID) -> dict[str, Any]:
    """The overlay merged and bound to `run_id` in `repo`, without writing a file."""
    base, overlay = repo / "configs" / "base.yaml", repo / "configs" / "overlay.yaml"
    merged = materialise(overlay, base)
    apply_run_identity(merged, run_id, f"results/raw/{run_id}")
    return merged


def _run_main(repo: Path, *, run_id: str = NEW_RUN_ID, out: str | None = None) -> tuple[int, Path]:
    """Run the CLI as a user would, returning the exit code and the path it was aimed at."""
    target = repo / (out if out is not None else f"results/raw/{run_id}/merged_config.yaml")
    code = materialise_eval_config.main(
        [
            "--overlay",
            "configs/overlay.yaml",
            "--base",
            "configs/base.yaml",
            "--run-id",
            run_id,
            "--out",
            str(target.relative_to(repo)),
        ]
    )
    return code, target


# --- 01-04: every identity field is rewritten -----------------------------------------


def test_01_a_stale_output_run_id_is_replaced(materialise_repo: Path) -> None:
    """The Phase 6I defect. The overlay still names the previous attempt; it must not survive."""
    merged = materialise(materialise_repo / "configs" / "overlay.yaml", materialise_repo / "configs" / "base.yaml")
    assert merged["output"]["run_id"] == STALE_RUN_ID, "the fixture must reproduce the inherited identity"

    apply_run_identity(merged, NEW_RUN_ID, NEW_RUN_DIR)

    assert merged["output"]["run_id"] == NEW_RUN_ID
    assert merged["output"]["run_id"] != STALE_RUN_ID


def test_02_output_rollouts_is_rewritten_under_the_run_directory(materialise_repo: Path) -> None:
    merged = _materialised(materialise_repo)

    assert merged["output"]["rollouts"] == f"{NEW_RUN_DIR}/rollouts.jsonl"
    assert STALE_RUN_ID not in merged["output"]["rollouts"]


def test_03_output_private_is_rewritten_under_the_run_directory(materialise_repo: Path) -> None:
    merged = _materialised(materialise_repo)

    assert merged["output"]["private"] == f"{NEW_RUN_DIR}/rollouts_private.jsonl"
    assert STALE_RUN_ID not in merged["output"]["private"]


def test_04_output_failures_is_rewritten_under_the_run_directory(materialise_repo: Path) -> None:
    merged = _materialised(materialise_repo)

    assert merged["output"]["failures"] == f"{NEW_RUN_DIR}/rollouts_failures.jsonl"
    assert STALE_RUN_ID not in merged["output"]["failures"]


# --- 05-08: the directory and the run id must agree -----------------------------------


def test_05_the_output_directory_basename_must_equal_the_run_id(materialise_repo: Path) -> None:
    """A config stored under one run's directory must not name another run."""
    code, target = _run_main(materialise_repo, run_id=NEW_RUN_ID, out=f"results/raw/{STALE_RUN_ID}/merged_config.yaml")

    assert code == 2
    assert not target.exists(), "a refused materialisation wrote a config"


def test_06_a_mismatched_run_id_and_out_directory_is_rejected(materialise_repo: Path) -> None:
    """Both directions: the id disagrees with a directory that is otherwise well-formed."""
    code, target = _run_main(
        materialise_repo, run_id=NEW_RUN_ID, out="results/raw/some_other_run_20261108T090000Z/merged_config.yaml"
    )

    assert code == 2
    assert not target.exists()


def test_07_an_out_path_outside_the_repository_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run directory with no repo-relative name cannot be named by any identity field."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(materialise_eval_config, "REPO_ROOT", root)
    _write_fixture(root)

    outside = tmp_path / "elsewhere" / f"results/raw/{NEW_RUN_ID}/merged_config.yaml"

    with pytest.raises(MaterialiseError, match="not inside the repository"):
        _run_dir_for(outside)

    assert not outside.exists()


def test_08_a_path_containing_updir_cannot_escape_the_run_directory(materialise_repo: Path) -> None:
    """`..` is resolved before containment is judged, not after.

    A config that writes `results/raw/<run>/../<other>/rollouts.jsonl` is claiming to live in
    `<run>` while writing into `<other>`. The identity check normalises the path first, so the
    escape is visible rather than hidden behind a parent that still matches the prefix.
    """
    merged = _materialised(materialise_repo)
    escaping = f"{NEW_RUN_DIR}/../{STALE_RUN_ID}/rollouts.jsonl"
    merged["output"]["rollouts"] = escaping

    reasons = run_identity_checks(
        cli_run_id=NEW_RUN_ID,
        manifest_run_id=NEW_RUN_ID,
        executed_run_id=merged["output"]["run_id"],
        config_run_id=merged["output"]["run_id"],
        run_dir=NEW_RUN_DIR,
        output_paths={"output.rollouts": escaping},
    )

    assert reasons, "a `..` escape was judged to be inside the run directory"
    assert any("output.rollouts is outside" in reason for reason in reasons), reasons
    # And the escape really does land elsewhere once resolved.
    assert posixpath.normpath(escaping).startswith(STALE_RUN_DIR)


# --- 09-12: the result is consistent, and consistently bound --------------------------


def test_09_a_materialised_config_passes_run_identity_checks(materialise_repo: Path) -> None:
    """The producer's output must satisfy the consumer's contract, unmodified."""
    code, target = _run_main(materialise_repo)

    assert code == 0, "materialisation refused a well-formed config"
    merged = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    output = merged["output"]

    reasons = run_identity_checks(
        cli_run_id=NEW_RUN_ID,
        manifest_run_id=NEW_RUN_ID,
        executed_run_id=output["run_id"],
        config_run_id=output["run_id"],
        run_dir=NEW_RUN_DIR,
        output_paths={
            "output.rollouts": output["rollouts"],
            "output.private": output["private"],
            "output.failures": output["failures"],
        },
    )

    assert reasons == []


def test_10_the_materialised_config_contains_no_stale_run_identity(materialise_repo: Path) -> None:
    """No field anywhere in the written file may still name the previous attempt."""
    code, target = _run_main(materialise_repo)
    assert code == 0

    text = target.read_text(encoding="utf-8")
    assert STALE_RUN_ID not in text, "a stale run identity survived materialisation"

    merged = yaml.safe_load(text) or {}
    output = merged["output"]
    assert output["run_id"] == NEW_RUN_ID
    for key in ("rollouts", "private", "failures"):
        assert NEW_RUN_ID in output[key], f"output.{key} does not name the new run"


def test_11_a_failed_materialisation_leaves_no_partial_output(materialise_repo: Path) -> None:
    """The refusal happens before the file is opened, so nothing half-written is left behind.

    A truncated config is worse than no config: the next step would fingerprint it and a
    manifest would record a file the run never executed.
    """
    code, target = _run_main(materialise_repo, run_id=NEW_RUN_ID, out=f"results/raw/{STALE_RUN_ID}/merged_config.yaml")

    assert code == 2
    assert not target.exists()
    # Nor a directory created on its way to refusing.
    assert not (materialise_repo / f"results/raw/{STALE_RUN_ID}").exists()


def test_12_output_analysis_may_remain_outside_the_raw_run_directory(materialise_repo: Path) -> None:
    """`output.analysis` is a summary, not a rollout trace, and lives under `results/analysis/`.

    It is therefore deliberately absent from the gate's `OUTPUT_PATH_FIELDS`: requiring it
    under the run directory would fail every correctly prepared run. This case pins both
    halves of that decision - where materialisation puts it, and that the gate tolerates it.
    """
    merged = _materialised(materialise_repo)

    assert merged["output"]["analysis"] == f"results/analysis/{NEW_RUN_ID}.json"
    assert not merged["output"]["analysis"].startswith(NEW_RUN_DIR)

    reasons = run_identity_checks(
        cli_run_id=NEW_RUN_ID,
        manifest_run_id=NEW_RUN_ID,
        executed_run_id=merged["output"]["run_id"],
        config_run_id=merged["output"]["run_id"],
        run_dir=NEW_RUN_DIR,
        # The gate is given all four; it must simply not complain about the analysis path.
        output_paths={
            "output.rollouts": merged["output"]["rollouts"],
            "output.private": merged["output"]["private"],
            "output.failures": merged["output"]["failures"],
            "output.analysis": merged["output"]["analysis"],
        },
    )

    assert reasons == []


# --- 13: the defect's evidence is untouched -------------------------------------------


#: The prepared run whose identity disagreed with its directory. Byte-frozen evidence.
FAILED_PREFLIGHT_ARTIFACTS = {
    f"results/raw/{STALE_RUN_ID}/manifest.json": "cf4e1624402702d7ac766dcbcbb929e38661593a8d84b484bfc42d808669e273",
    "results/raw/phase6i_bugfix_offbyone_pagination_20261106T162531Z/manifest.json": (
        "beaf9c6c9e3844f903f71d6cc780fd3eeb336e8ddefab3670d0e19ec5a69396c"
    ),
    "results/raw/phase6i_bugfix_offbyone_pagination_20261107T040554Z/manifest.json": (
        "7bdf65c04a69ad207d5d494acb633256626afbff1632a11c3acee3a2b109bf96"
    ),
    "results/raw/phase6i_bugfix_offbyone_pagination_20261107T040554Z/merged_config.yaml": (
        "7e295c2721a21fb0adf757f1cd1d08efd7619f5930d2c30a90bf63236abbac16"
    ),
    "results/raw/phase6i_bugfix_offbyone_pagination_20261107T040554Z/vllm_server.log": (
        "9d42f7ecf17faa8333282a57fdfb2344ece403bd3b30dac1efb392bf6e7dd437"
    ),
}


def test_13_failed_preflight_evidence_is_byte_identical() -> None:
    """Materialisation is a pure producer: running it must not disturb existing evidence.

    Checked against the committed files rather than the fixtures above, because a fix that
    quietly rewrote a failed run's config would erase the only record of what went wrong.
    """
    missing = [rel for rel in FAILED_PREFLIGHT_ARTIFACTS if not (REPO_ROOT / rel).is_file()]
    if missing:
        pytest.skip(f"evidence absent, nothing to preserve: {missing}")

    for rel, expected in FAILED_PREFLIGHT_ARTIFACTS.items():
        digest = hashlib.sha256((REPO_ROOT / rel).read_bytes()).hexdigest()
        assert digest == expected, f"{rel} was modified"
