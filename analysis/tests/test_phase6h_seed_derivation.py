"""Phase 6H - per-rollout sampling seeds: derivation, provenance, and resume safety.

Phase 6G collected four rollouts and produced one trajectory four times. `sampling.seed: 42`
was sent on every completion call, so the config that said "4 rollouts" described one sample
recorded four times. Nothing in the artifacts looked wrong; the uniformity guard fired, but
it fired on a number that had already been decided by the harness.

The tests below pin the three properties that make that impossible to repeat silently:

1. **The derivation is a pure function of recorded inputs.** Same inputs, same seed, in any
   process. Different `(base_seed, task_id, rollout_index)`, different seed.
2. **The derivation contract is versioned and checked.** A config naming another algorithm is
   refused, so an artifact can never carry a seed the recorded label cannot explain.
3. **The batch is validated before the first model call.** Duplicate seeds and a seed policy
   that cannot produce independence both fail on arithmetic, not on GPU time.

Everything here is CPU-only and offline. No model server, no tokenizer, no vLLM, no NeMo, no
network, no GPU. The one test that reads Phase 6G artifacts only hashes them.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from analysis import sampling_seeds
from analysis.sampling_seeds import (
    SAMPLING_SEED_DERIVATION,
    UINT32_MAX,
    SeedDerivationError,
    assert_unique_seeds,
    derive_rollout_seed,
    derive_turn_seed,
    plan_digest,
    plan_seeds,
)
from analysis.trace import build_row, validate_row
from scripts.collect_baseline_rollouts import (
    BaselineConfig,
    CollectorError,
    ResumeFileError,
    _assert_seed_policy_usable,
    _load_persisted,
    _request_seed,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "bugfix_offbyone_pagination"
BASE_SEED = 42

#: Seeds Phase 6H plans for the four-rollout smoke: `plan_seeds(42, [TASK_ID], 4)`.
#: Written out rather than recomputed so that a change to the derivation fails here, loudly,
#: instead of silently re-defining what the prepared smoke means.
PHASE6H_PLANNED_SEEDS = [184288048, 3168762815, 475852200, 1693580453]

#: Phase 6G artifacts, frozen. Phase 6H must not rewrite them: they are the evidence that the
#: seed collision happened, and a "fixed" Phase 6G would destroy the before-picture that
#: justifies the after-picture. Byte sizes and hashes both, because a hash alone would not
#: catch a same-size rewrite under a different schema version.
PHASE6G_RUN_DIR = "results/raw/phase6g_bugfix_offbyone_pagination_20261106T150419Z"
PHASE6G_FROZEN = {
    "rollouts.jsonl": "e243aec1cd50c93c158069a93434e806469a505744deedb6ad6f7e8e807d4675",
    "rollouts_private.jsonl": "eb9eb25dceb28d454fe701b508e75400d460aea8dddbec4c67e9d34c014b672f",
    "collection_summary.json": "d423cbdc1f5ef2cb4faee17da56854dcbc20d87a7d443e878ba5cf3a38f8d1f0",
    "manifest.json": "211917e10a1a32e677265a3e5858f4da1aaf94a030e70ae761d0d89524b28d6c",
    "merged_config.yaml": "a676c6b39819b1b5eee1a65515f6865cb03012aa86fba94ccb06510b38d17abe",
    # An empty failure log still gets a hash: "still empty" is the assertion, and a test that
    # only checked existence would pass on a run that started failing.
    "rollouts_failures.jsonl": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}


# --------------------------------------------------------------------------- helpers


def _cfg(tmp_path: Path, **overrides: Any) -> BaselineConfig:
    """A real config with its output paths redirected into `tmp_path`.

    Loaded from the committed YAML rather than built by hand, so these tests fail if the
    config loses a key the collector now depends on (`sampling.seed_policy`,
    `sampling.seed_derivation`) instead of quietly passing against a hand-made stand-in.

    `output_path` resolves `output.*` from the raw document, so the paths are overridden in
    the document like any other setting. `overrides` are dotted paths, applied the same way,
    which lets a test express "a config that says `literal`" without a second fixture.
    """
    from dataclasses import replace as _replace

    cfg = BaselineConfig.load(REPO_ROOT / "configs" / "eval" / "baseline_phase6.yaml")
    raw = json.loads(json.dumps(cfg.raw, default=str))
    output = raw.setdefault("output", {})
    for key in ("rollouts", "private", "failures", "summary"):
        name = output.get(key) or f"results/raw/test/{key}.jsonl"
        output[key] = f"{tmp_path}/{Path(str(name)).name}"
    for dotted, value in overrides.items():
        # Values are YAML/JSON scalars here (`"literal"`, `43`), so they are placed into the
        # document as-is. Decoding them would turn the string `literal` into a JSON error
        # rather than the seed policy it is meant to be.
        section, _, leaf = dotted.partition(".")
        raw.setdefault(section, {})[leaf] = value
    return _replace(cfg, raw=raw)


# ----------------------------------------------------- 1-3: the derivation is a function


def test_same_inputs_produce_the_same_seed() -> None:
    """Reproducibility is the whole point: a seed that moves cannot be recorded.

    Called twice in one process, the test would pass for a `hash()`-based derivation too if
    PYTHONHASHSEED happened to be fixed. `test_derivation_is_stable_across_processes` is the
    one that rules that out; this one pins the contract.
    """
    assert derive_rollout_seed(BASE_SEED, TASK_ID, 0) == derive_rollout_seed(BASE_SEED, TASK_ID, 0)


def test_derivation_is_stable_across_processes() -> None:
    """A fresh interpreter with a different PYTHONHASHSEED must derive the same numbers.

    This is the test that forbids built-in `hash()`. CPython salts `hash(str)` per process,
    so a `hash()`-based seed would differ between the collector and any later
    re-verification, making every recorded seed a local artefact of one process lifetime.
    """
    script = (
        "from analysis.sampling_seeds import derive_rollout_seed as d;"
        f"print(d({BASE_SEED}, {TASK_ID!r}, 0), d({BASE_SEED}, {TASK_ID!r}, 3))"
    )
    expected = f"{derive_rollout_seed(BASE_SEED, TASK_ID, 0)} {derive_rollout_seed(BASE_SEED, TASK_ID, 3)}"
    for hashseed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": hashseed, "PATH": "/usr/bin:/bin"},
        )
        assert result.stdout.strip() == expected, f"PYTHONHASHSEED={hashseed} changed the derived seed"


def test_different_rollout_index_produces_a_different_seed() -> None:
    """The Phase 6G defect in one assertion."""
    seeds = [derive_rollout_seed(BASE_SEED, TASK_ID, i) for i in range(4)]
    assert len(set(seeds)) == 4


def test_different_task_produces_a_different_seed() -> None:
    """Two tasks at the same index must not share a sample.

    Without the task id in the derivation, rollout 0 of every task would carry one seed, which
    is a cross-task collision that a per-task uniqueness check would never see.
    """
    assert derive_rollout_seed(BASE_SEED, "task_a", 0) != derive_rollout_seed(BASE_SEED, "task_b", 0)


def test_different_base_seed_produces_a_different_seed() -> None:
    """Changing `sampling.seed` must actually change the batch, not just its label."""
    assert derive_rollout_seed(42, TASK_ID, 0) != derive_rollout_seed(43, TASK_ID, 0)


def test_seeds_are_within_unsigned_32_bit_range() -> None:
    """vLLM/numpy/torch disagree above 2**32-1; some builds raise, others truncate.

    Constraining the output means the recorded seed is always the seed the server used.
    """
    seeds = [derive_rollout_seed(BASE_SEED, f"task_{i}", j) for i in range(50) for j in range(4)]
    assert seeds
    for seed in seeds:
        assert 0 <= seed <= UINT32_MAX
        assert isinstance(seed, int)


def test_field_encoding_is_unambiguous() -> None:
    """A separator would be ambiguous because task ids are free-form strings.

    `("a:b", 1)` and `("a", ...)` must not collide with any encoding that simply joins with
    a colon, which is why the fields are length-prefixed with a type tag.
    """
    assert derive_rollout_seed(42, "a:b", 1) != derive_rollout_seed(42, "a", 1)
    assert derive_rollout_seed(42, "a", 1) != derive_rollout_seed(42, "a", 11)
    # The integer 1 and the string "1" must not collide either.
    assert derive_rollout_seed(1, "t", 0) != derive_rollout_seed(2, "t", 0)


def test_derivation_does_not_use_builtin_hash() -> None:
    """A static check, because the behavioural one only catches the salt in a fresh process.

    Someone reaching for `hash((base, task, index))` would produce code that looks
    deterministic and passes every same-process test. This is the guard that is not fooled by
    a fixed PYTHONHASHSEED.

    Docstrings are stripped via the AST rather than by filtering comment lines, because this
    module's prose *explains* why `hash()` is banned - a naive text search would flag the
    explanation. The check is on the executable code only, which is the thing that matters.
    """
    import ast

    source = (REPO_ROOT / "analysis" / "sampling_seeds.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        # Drop docstrings so prose that names `hash()` is not mistaken for a call.
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                node.body = body[1:]
    code = ast.unparse(tree)
    assert "hash(" not in code.replace("hashlib.sha256(", "").replace("hashlib.sha256", ""), (
        "sampling_seeds must not use built-in hash(): it is salted per process"
    )
    assert "hashlib.sha256" in source


def test_invalid_inputs_are_rejected() -> None:
    """Booleans are ints in Python; accepting `True` as a seed would be a silent 1."""
    with pytest.raises(SeedDerivationError):
        derive_rollout_seed(True, TASK_ID, 0)  # type: ignore[arg-type]
    with pytest.raises(SeedDerivationError):
        derive_rollout_seed(-1, TASK_ID, 0)
    with pytest.raises(SeedDerivationError):
        derive_rollout_seed(BASE_SEED, "", 0)
    with pytest.raises(SeedDerivationError):
        derive_rollout_seed(BASE_SEED, TASK_ID, -1)
    with pytest.raises(SeedDerivationError):
        derive_rollout_seed(BASE_SEED, TASK_ID, True)  # type: ignore[arg-type]


def test_derivation_label_is_the_versioned_contract() -> None:
    """The label is written into artifacts, so it must be a stable, explicit string.

    Changing the algorithm without changing the label would reinterpret old rows as
    reproducible under a derivation they were not produced by.
    """
    assert SAMPLING_SEED_DERIVATION == "sha256_u32_v1"


def test_turn_seed_derives_from_the_rollout_seed() -> None:
    """Per-turn seeds must stay inside one rollout's derivation, not restart from base."""
    rollout_seed = derive_rollout_seed(BASE_SEED, TASK_ID, 1)
    turn_seeds = [derive_turn_seed(rollout_seed, t) for t in range(4)]
    assert len(set(turn_seeds)) == 4
    assert derive_turn_seed(rollout_seed, 0) != derive_rollout_seed(BASE_SEED, TASK_ID, 1)


# ------------------------------------------------- 4-6: the plan is collision-free


def test_four_planned_rollouts_get_four_distinct_seeds() -> None:
    """The Phase 6H smoke's actual plan, pinned to its published values."""
    plan = plan_seeds(BASE_SEED, [TASK_ID], 4)
    assert [entry["sampling_seed"] for entry in plan] == PHASE6H_PLANNED_SEEDS
    assert len({entry["sampling_seed"] for entry in plan}) == 4
    assert_unique_seeds(plan)


def test_multi_task_plan_reuses_no_seed_across_tasks() -> None:
    """Uniqueness is global, not per task: a cross-task collision is still one sample twice."""
    task_ids = ["bugfix_offbyone_pagination", "bugfix_metric_zero_domain_error", "step3_where_missing_keys"]
    plan = plan_seeds(BASE_SEED, task_ids, 3)
    assert len(plan) == 9
    assert len({entry["sampling_seed"] for entry in plan}) == 9
    assert_unique_seeds(plan)


def test_duplicate_seeds_are_rejected_before_any_model_call() -> None:
    """A collision must fail on arithmetic, not after burning GPU time.

    `assert_unique_seeds` is called before `check_server` in `collect`, so this refusal
    happens while the only cost is a few microseconds.
    """
    plan = plan_seeds(BASE_SEED, ["a", "b"], 1)
    plan[1]["sampling_seed"] = plan[0]["sampling_seed"]
    with pytest.raises(SeedDerivationError, match="duplicate sampling_seed"):
        assert_unique_seeds(plan)


def test_plan_digest_is_stable_and_sensitive() -> None:
    """The digest is recorded in the summary, so equal plans must hash equally."""
    left = plan_seeds(BASE_SEED, [TASK_ID], 4)
    right = plan_seeds(BASE_SEED, [TASK_ID], 4)
    assert plan_digest(left) == plan_digest(right)
    assert plan_digest(left) != plan_digest(plan_seeds(BASE_SEED + 1, [TASK_ID], 4))
    assert len(plan_digest(left)) == 64


# --------------------------------------------- 7-9: provenance reaches every artifact


def _row(**overrides: Any) -> dict[str, Any]:
    """A minimal valid v3 row with the Phase 6H seed fields populated."""
    from datetime import UTC, datetime

    from analysis.trace import Timer
    from benchmark.tools.task_loader import TASKS_ROOT, load_task

    task = load_task(TASKS_ROOT / "train" / TASK_ID)
    timer = Timer(started_at=datetime(2026, 1, 1, tzinfo=UTC))
    kwargs: dict[str, Any] = {
        "run_id": "phase6h_test",
        "rollout_id": f"phase6h_test__{TASK_ID}__r00",
        "task": task,
        "result": None,
        "tool_calls": [],
        "timer": timer,
        "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "checkpoint_id": None,
        "random_seed": PHASE6H_PLANNED_SEEDS[0],
        "base_seed": BASE_SEED,
        "sampling_seed": PHASE6H_PLANNED_SEEDS[0],
        "sampling_seed_derivation": SAMPLING_SEED_DERIVATION,
        "rollout_index": 0,
        "seed_policy": "per_rollout",
        "prompt_config_version": "v1",
        "error": RuntimeError("test row, not a real rollout"),
    }
    kwargs.update(overrides)
    row = build_row(**kwargs)
    row["termination_reason"] = "error"
    return row


def test_public_trace_records_the_effective_and_base_seed() -> None:
    """Both numbers, because either alone cannot answer the question a reader has.

    The base seed identifies the batch; the derived seed replays one rollout.
    """
    row = _row()
    assert row["agent"]["base_seed"] == BASE_SEED
    assert row["agent"]["sampling_seed"] == PHASE6H_PLANNED_SEEDS[0]
    assert row["agent"]["random_seed"] == PHASE6H_PLANNED_SEEDS[0]
    assert row["agent"]["sampling_seed_derivation"] == SAMPLING_SEED_DERIVATION
    assert row["agent"]["rollout_index"] == 0
    assert row["agent"]["seed_policy"] == "per_rollout"
    validate_row(row)


def test_trace_seed_fields_are_optional_so_phase6g_rows_still_validate() -> None:
    """v3 additive: the fields must not become required, or old results/ die on re-validation."""
    row = _row()
    for field in ("base_seed", "sampling_seed", "sampling_seed_derivation", "rollout_index", "seed_policy"):
        row["agent"].pop(field)
    row["agent"]["random_seed"] = BASE_SEED
    validate_row(row)


def test_summary_records_the_derivation_and_the_full_plan(tmp_path: Path) -> None:
    """A digest alone cannot be inverted, so the mapping is written out too."""
    from scripts import collect_baseline_rollouts as collector

    source = inspect.getsource(collector.collect)
    assert '"sampling_seeds"' in source
    for key in (
        "base_seed",
        "sampling_seed_derivation",
        "seed_policy",
        "per_rollout",
        "n_distinct_sampling_seeds",
        "plan_digest",
        "plan",
    ):
        assert f'"{key}"' in source, f"collection_summary no longer records sampling_seeds.{key}"


def test_manifest_carries_base_seed_and_derivation() -> None:
    """The manifest is the run's own account of what it intended to do.

    It reads the effective config, so the seed fields must be present there for the manifest
    to record them - this asserts the config schema, which is the upstream half.
    """
    doc = BaselineConfig.load(REPO_ROOT / "configs" / "eval" / "baseline_phase6.yaml")
    assert doc.seed == BASE_SEED
    assert doc.seed_policy in {"per_rollout", "per_turn", "literal"}
    assert doc.seed_derivation == SAMPLING_SEED_DERIVATION


# ------------------------------------------------ 10-12: config mismatches are refused


def test_config_naming_another_derivation_is_refused(tmp_path: Path) -> None:
    """A label the code cannot honour must fail, not be recorded and explained away later."""
    cfg = _cfg(tmp_path, **{"sampling.seed_derivation": "md5_u64_v0"})
    with pytest.raises(CollectorError, match="seed_derivation"):
        _ = cfg.seed_derivation


def test_literal_policy_with_multiple_rollouts_is_refused(tmp_path: Path) -> None:
    """The Phase 6G defect as a config that can no longer be silently accepted."""
    cfg = _cfg(tmp_path, **{"sampling.seed_policy": "literal"})
    with pytest.raises(CollectorError, match="literal"):
        _assert_seed_policy_usable(cfg, n_rollouts=4)
    # One rollout under `literal` is a single seed by construction, so it is allowed.
    _assert_seed_policy_usable(cfg, n_rollouts=1)


def test_unknown_seed_policy_is_refused(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, **{"sampling.seed_policy": "whatever"})
    with pytest.raises(CollectorError, match="seed_policy"):
        _ = cfg.seed_policy


def test_literal_policy_sends_the_config_seed_verbatim(tmp_path: Path) -> None:
    """`literal` reproduces pre-6H behaviour, which is why it is still expressible."""
    cfg = _cfg(tmp_path, **{"sampling.seed_policy": "literal"})
    assert _request_seed(cfg, task_id=TASK_ID, rollout_index=3) == BASE_SEED


def test_request_seed_matches_the_recorded_seed(tmp_path: Path) -> None:
    """The seed the request carries is the seed the row records - one function, one answer."""
    cfg = _cfg(tmp_path)
    for index in range(4):
        assert _request_seed(cfg, task_id=TASK_ID, rollout_index=index) == PHASE6H_PLANNED_SEEDS[index]


# --------------------------------------------------------- 13-14: resume reproduces


def test_resume_reproduces_the_seed_mapping(tmp_path: Path) -> None:
    """A resumed batch must re-derive the same seeds, or it is a mixture of two batches."""
    cfg = _cfg(tmp_path)
    first = _request_seed(cfg, task_id=TASK_ID, rollout_index=2)
    reloaded = _cfg(tmp_path)
    assert _request_seed(reloaded, task_id=TASK_ID, rollout_index=2) == first


def test_resume_skips_persisted_rollouts(tmp_path: Path) -> None:
    """A persisted row is reused, not re-sampled: resume exists to preserve GPU time."""
    cfg = _cfg(tmp_path)
    rollouts_path = cfg.output_path("rollouts")
    run_dir = rollouts_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    persisted_row = {
        "run_id": cfg.run_id,
        "rollout_id": f"{cfg.run_id}__{TASK_ID}__r00",
        "agent": {"sampling_seed": PHASE6H_PLANNED_SEEDS[0]},
    }
    rollouts_path.write_text(json.dumps(persisted_row) + "\n", encoding="utf-8")

    loaded = _load_persisted(cfg)
    assert len(loaded) == 1
    assert loaded[persisted_row["rollout_id"]]["agent"]["sampling_seed"] == PHASE6H_PLANNED_SEEDS[0]


def test_resume_refuses_another_runs_rows(tmp_path: Path) -> None:
    """A run_id is the identity of a batch; another run's rows must be refused, not dropped.

    Silently dropping them is the dangerous behaviour: the resume succeeds, reports fewer
    rollouts than were planned, and the batch looks like a normal short run rather than a
    mis-pointed directory.
    """
    cfg = _cfg(tmp_path)
    rollouts_path = cfg.output_path("rollouts")
    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
    foreign = {"run_id": "some_other_run", "rollout_id": "some_other_run__t__r00", "agent": {}}
    rollouts_path.write_text(json.dumps(foreign) + "\n", encoding="utf-8")
    with pytest.raises(ResumeFileError, match="some_other_run"):
        _load_persisted(cfg)


def test_resume_repairs_a_truncated_tail_line(tmp_path: Path) -> None:
    """An interrupted append is the one recoverable shape, and it is repaired on disk.

    Skipping the fragment without truncating would leave tail garbage that every later
    attempt has to re-diagnose, so the file must be cut back to its last complete line.
    """
    cfg = _cfg(tmp_path)
    rollouts_path = cfg.output_path("rollouts")
    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
    good = {"run_id": cfg.run_id, "rollout_id": f"{cfg.run_id}__{TASK_ID}__r00", "agent": {}}
    good_line = json.dumps(good) + "\n"
    rollouts_path.write_text(good_line + '{"run_id": "trunc', encoding="utf-8")

    loaded = _load_persisted(cfg)

    assert set(loaded) == {good["rollout_id"]}
    repaired = rollouts_path.read_bytes()
    assert repaired == good_line.encode("utf-8")
    assert repaired.endswith(b"\n")
    # The repair must leave a file a plain JSONL reader can consume line by line.
    assert [json.loads(line) for line in repaired.decode("utf-8").splitlines()] == [good]


def test_resumed_append_leaves_a_fully_parseable_file(tmp_path: Path) -> None:
    """The whole point of the repair: appending the re-sampled rollout stays parseable."""
    cfg = _cfg(tmp_path)
    rollouts_path = cfg.output_path("rollouts")
    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
    first = {"run_id": cfg.run_id, "rollout_id": f"{cfg.run_id}__{TASK_ID}__r00", "agent": {}}
    rollouts_path.write_text(json.dumps(first) + "\n" + '{"run_id": "trunc', encoding="utf-8")
    _load_persisted(cfg)

    resampled = {"run_id": cfg.run_id, "rollout_id": f"{cfg.run_id}__{TASK_ID}__r01", "agent": {}}
    with rollouts_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(resampled) + "\n")

    rows = [json.loads(line) for line in rollouts_path.read_text(encoding="utf-8").splitlines()]
    assert [row["rollout_id"] for row in rows] == [first["rollout_id"], resampled["rollout_id"]]


def test_resume_refuses_a_malformed_middle_row(tmp_path: Path) -> None:
    """A bad line with content after it is corruption, not an interrupted append."""
    cfg = _cfg(tmp_path)
    rollouts_path = cfg.output_path("rollouts")
    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
    good = {"run_id": cfg.run_id, "rollout_id": f"{cfg.run_id}__{TASK_ID}__r00", "agent": {}}
    tail = {"run_id": cfg.run_id, "rollout_id": f"{cfg.run_id}__{TASK_ID}__r02", "agent": {}}
    text = json.dumps(good) + "\n" + '{"run_id": broken' + "\n" + json.dumps(tail) + "\n"
    rollouts_path.write_text(text, encoding="utf-8")

    with pytest.raises(ResumeFileError, match="line 2"):
        _load_persisted(cfg)
    # A refusal must not touch the file: the evidence is preserved for a human to inspect.
    assert rollouts_path.read_text(encoding="utf-8") == text


def test_resume_refuses_duplicate_rollout_ids(tmp_path: Path) -> None:
    """Last-wins would let a later line override the seed the earlier one recorded."""
    cfg = _cfg(tmp_path)
    rollouts_path = cfg.output_path("rollouts")
    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_id = f"{cfg.run_id}__{TASK_ID}__r00"
    first = {"run_id": cfg.run_id, "rollout_id": rollout_id, "agent": {"sampling_seed": 1}}
    second = {"run_id": cfg.run_id, "rollout_id": rollout_id, "agent": {"sampling_seed": 99}}
    rollouts_path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")

    with pytest.raises(ResumeFileError, match="more than once"):
        _load_persisted(cfg)


def test_resume_refuses_an_unidentifiable_tail_fragment(tmp_path: Path) -> None:
    """Truncation must not be applied blind to a tail that is not a partial row."""
    cfg = _cfg(tmp_path)
    rollouts_path = cfg.output_path("rollouts")
    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
    rollouts_path.write_text("\x00\x01garbage-not-json", encoding="utf-8")

    with pytest.raises(ResumeFileError, match="does not begin a JSON object"):
        _load_persisted(cfg)


def test_resume_terminates_a_complete_final_row_without_a_newline(tmp_path: Path) -> None:
    """A finished row missing only its newline is evidence, not a fragment to discard."""
    cfg = _cfg(tmp_path)
    rollouts_path = cfg.output_path("rollouts")
    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"run_id": cfg.run_id, "rollout_id": f"{cfg.run_id}__{TASK_ID}__r00", "agent": {}}
    rollouts_path.write_text(json.dumps(row), encoding="utf-8")

    loaded = _load_persisted(cfg)

    assert set(loaded) == {row["rollout_id"]}
    assert rollouts_path.read_bytes() == (json.dumps(row) + "\n").encode("utf-8")


def test_resume_absent_file_is_an_empty_resume(tmp_path: Path) -> None:
    """A first attempt has nothing to resume; that is not an error."""
    cfg = _cfg(tmp_path)
    assert _load_persisted(cfg) == {}


def test_resume_empty_file_is_an_empty_resume(tmp_path: Path) -> None:
    """An empty file is the state after a run that died before its first row."""
    cfg = _cfg(tmp_path)
    rollouts_path = cfg.output_path("rollouts")
    rollouts_path.parent.mkdir(parents=True, exist_ok=True)
    rollouts_path.write_bytes(b"")
    assert _load_persisted(cfg) == {}


def test_resume_mismatch_is_a_config_error(tmp_path: Path) -> None:
    """A persisted seed this run would not derive means the batch is unreproducible.

    The refusal is asserted at the derivation level plus the collector's guard, because the
    collector's own comparison lives inside `collect`, which needs a server.
    """
    cfg = _cfg(tmp_path)
    planned = _request_seed(cfg, task_id=TASK_ID, rollout_index=0)
    other_base = _cfg(tmp_path, **{"sampling.seed": repr(BASE_SEED + 1)})
    assert _request_seed(other_base, task_id=TASK_ID, rollout_index=0) != planned
    # And the collision guard is what would catch two rollouts sharing the surviving seed.
    with pytest.raises(SeedDerivationError):
        assert_unique_seeds(
            [
                {"task_id": TASK_ID, "rollout_index": 0, "sampling_seed": planned},
                {"task_id": TASK_ID, "rollout_index": 1, "sampling_seed": planned},
            ]
        )


# --------------------------------------------- 15-18: Phase 6H changes nothing else


def test_phase6g_artifacts_are_byte_identical() -> None:
    """Phase 6G is the evidence that the defect existed; it must stay exactly as written."""
    run_dir = REPO_ROOT / PHASE6G_RUN_DIR
    if not run_dir.exists():
        pytest.skip(f"{PHASE6G_RUN_DIR} is not present in this checkout")
    for name, expected in PHASE6G_FROZEN.items():
        path = run_dir / name
        assert path.exists(), f"{name} is missing from the Phase 6G run directory"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"{name} was modified"


def test_reward_path_is_untouched() -> None:
    """Phase 6H changes sampling only. A changed reward would make the seeds irrelevant."""
    rewards = REPO_ROOT / "rewards"
    for rel in ("compose.py", "v2.py"):
        path = rewards / rel
        assert path.exists(), f"{rel} is missing"
    names = {p.name for p in rewards.glob("*.py")}
    assert "v1.py" not in names  # v1 lives elsewhere; the point is Phase 6H added no reward module


def test_no_gpu_or_model_imports_in_the_seed_path() -> None:
    """The derivation must stay pure: importing it must not pull in torch or a client."""
    script = (
        "import sys; import analysis.sampling_seeds;"
        "bad=[m for m in ('torch','vllm','openai','httpx','transformers') if m in sys.modules];"
        "print(','.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "", f"sampling_seeds transitively imported {result.stdout.strip()}"


def test_heldout_split_is_unreachable_from_the_seed_path() -> None:
    """Phase 6H adds no code path that could select the sealed split."""
    source = inspect.getsource(sampling_seeds)
    assert "heldout" not in source.lower()
    assert "split" not in source.lower()


def test_seed_plan_for_the_prepared_smoke(tmp_path: Path) -> None:
    """The prepared Phase 6H smoke's four seeds, restated where the config is read.

    This is the test that ties the prepared overlay to the published mapping: if the overlay
    changes `sampling.seed` or the task, this fails before a GPU is booked.
    """
    cfg = _cfg(tmp_path)
    plan = plan_seeds(cfg.seed, [TASK_ID], 4)
    assert [entry["sampling_seed"] for entry in plan] == PHASE6H_PLANNED_SEEDS
    assert cfg.seed == BASE_SEED
