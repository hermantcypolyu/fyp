"""Regression tests for the five defects the Phase 6E diagnostic smoke exposed.

Phase 6E was not wasted: four rollouts ran, and the artifacts they left behind showed that
several of the collector's guarantees were stated in comments but not in code. Each test
below pins one of them, so the claim cannot quietly rot back into a comment.

The defects, in the order Phase 6F fixed them:

1. The uniform-reward guard raised *before* persistence, so the rejected batch - the one
   most worth reading - was destroyed by the guard that rejected it.
2. `--diagnostic-only` and `--tokenizer` assigned to a frozen dataclass and crashed.
3. `--diagnostic-only` was a label on the output, not a veto on the verdict.
4. `context_budget` was written in a shape the schema does not accept, and to one file only.
5. The manifest reported the overlay's *defaults* as if they were the run's settings.

Everything here is CPU-only. Nothing contacts a model server, loads a tokenizer, or touches
a GPU: the tests build the objects the collector builds and inspect them.
"""

from __future__ import annotations

import json
import stat
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from analysis.trace import PRIVATE_MODE, TraceError, secure_write_private_log
from scripts.collect_baseline_rollouts import (
    BaselineConfig,
    CollectorError,
    ContextBudget,
    Probe,
    _assess_validity,
    _persist_evidence,
    _public_context_telemetry,
    _with_overrides,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- helpers


def _probe(prompt_tokens: int | None = 1024, *, status: str = "ok") -> Probe:
    return Probe(
        prompt_tokens=prompt_tokens,
        budget=ContextBudget(max_model_len=32768, reserved_completion_tokens=4096, safety_margin_tokens=512),
        status=status,
    )


def _cfg_at(tmp_path: Path, **overrides: Any) -> BaselineConfig:
    """A config whose output paths point into `tmp_path`.

    `output_path` resolves `output.*` from the raw document, so the paths are overridden in
    the document like any other setting - which also exercises `_with_overrides`.
    """
    cfg = BaselineConfig.load()
    raw = json.loads(json.dumps(cfg.raw, default=str))
    output = raw.setdefault("output", {})
    for key in ("rollouts", "private", "failures", "summary"):
        name = output.get(key) or f"results/raw/test/{key}.jsonl"
        output[key] = f"{tmp_path}/{Path(str(name)).name}"
    cfg = replace(cfg, raw=raw)
    return _with_overrides(cfg, **overrides) if overrides else cfg


def _summary(**overrides: Any) -> dict[str, Any]:
    """A summary shaped like the collector's, enough for `_assess_validity` to judge."""
    values: dict[str, Any] = {
        "tasks_with_zero_valid_rollouts": [],
        "n_context_overflow": 0,
        "n_reward_eligible": 4,
    }
    values.update(overrides)
    return values


def _rewards() -> list[float]:
    return [0.0, 0.5, 1.0, 0.0]


def _row(rollout_id: str = "r0", *, reward: float = 0.0, termination: str = "completed") -> dict[str, Any]:
    """A real v3 trace row, built by the same function the collector uses.

    The collector's public write path validates every row against the schema, so a
    plausible-looking dict would test nothing: it would fail on `schema_version` before
    reaching the behaviour under test. `build_row` needs a Task and a VerifyResult, both of
    which are plain dataclasses, so constructing them costs nothing and keeps this test
    honest about what shape the collector writes.
    """
    from analysis.trace import Timer, build_row
    from benchmark.tools.task_loader import load_task
    from envs.coding_env.verifier import VerifyResult

    # A real task from the train split: the schema validates category and difficulty
    # against closed enums, so a synthetic dict would have to guess at them, and a guess
    # that drifted from the corpus would make this test lie about what a real row contains.
    task = load_task(REPO_ROOT / "benchmark" / "tasks" / "train" / "bugfix_dotenv_bom_prefix")
    result = VerifyResult(
        task_id="noop",
        reward=reward,
        pass_fraction=reward,
        passed=0,
        collected=0,
        submitted=termination == "submitted",
        # The schema requires the two caps every host can report; recording them as
        # actually applied is what a real graded episode says.
        limits_enforced={"wall_clock_s": True, "network": True},
    )
    timer = Timer.start()
    return build_row(
        run_id="phase6f-test",
        rollout_id=rollout_id,
        task=task,
        result=result,
        tool_calls=[],
        timer=timer,
        validate=True,
    )


# ------------------------------------------------------- defect 1: persist, then judge


def test_uniform_batch_is_persisted_before_the_guard_can_raise(tmp_path: Path) -> None:
    """The guard must never be the thing that destroys the evidence it objects to.

    Phase 6E's guard raised first and left nothing on disk. A batch that fails acceptance is
    the batch a human most wants to read, so persistence comes first and is measured from
    disk afterwards.
    """
    cfg = _cfg_at(tmp_path)
    rows = [_row(f"r{i}", reward=0.0) for i in range(4)]
    private = [{"rollout_id": f"r{i}", "tool_calls": []} for i in range(4)]

    persisted = _persist_evidence(cfg, rows, private, _summary())

    assert persisted["persisted"] is True
    assert persisted["n_public_rows"] == 4
    assert persisted["n_private_rows"] == 4
    assert Path(persisted["rollouts_path"]).is_file()
    assert Path(persisted["private_path"]).is_file()


def test_persistence_counts_are_read_back_from_disk(tmp_path: Path) -> None:
    """`evidence_persisted` is a claim about disk, so it is measured from disk."""
    cfg = _cfg_at(tmp_path)
    persisted = _persist_evidence(
        cfg,
        [_row("r0")],
        [{"rollout_id": "r0"}],
        _summary(),
    )
    assert persisted["private_mode"] == f"{PRIVATE_MODE:04o}"
    assert len(Path(persisted["rollouts_path"]).read_text(encoding="utf-8").splitlines()) == 1


def test_a_row_count_mismatch_is_an_error_not_a_warning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A sidecar describing a different number of rollouts cannot be joined to the public rows.

    Zero private rows for zero collected private records is consistent and must NOT raise;
    the failure this guards against is the log silently losing rows on the way to disk, so
    the writer is made to drop one here.
    """
    import scripts.collect_baseline_rollouts as collector_module

    # Patched where the collector looks the name up, not where it was defined: the module
    # imported the function directly, so patching `analysis.trace` would change nothing.
    monkeypatch.setattr(
        collector_module,
        "secure_write_private_log",
        lambda path, records, **kw: path.write_text(""),
    )
    with pytest.raises(CollectorError):
        _persist_evidence(
            _cfg_at(tmp_path),
            [_row("r0")],
            [{"rollout_id": "r0"}],
            _summary(),
        )


# ------------------------------------------------- defect 2: overrides on a frozen config


def test_overrides_do_not_mutate_the_loaded_config() -> None:
    """The config is frozen so that "what this run used" cannot change under a reader."""
    cfg = BaselineConfig.load()
    assert cfg.diagnostic_only is False

    overridden = _with_overrides(cfg, diagnostic_only=True)

    assert overridden.diagnostic_only is True
    assert cfg.diagnostic_only is False, "the original object must be untouched"
    assert overridden is not cfg


def test_config_remains_frozen_after_an_override() -> None:
    """`dataclasses.replace` keeps the frozen property; making the dataclass mutable was the wrong fix."""
    cfg = _with_overrides(BaselineConfig.load(), diagnostic_only=True)
    with pytest.raises(FrozenInstanceError):
        cfg.diagnostic_only = False  # type: ignore[misc]


def test_tokenizer_override_reaches_the_config_it_claims_to_change() -> None:
    """The flag existed in Phase 6E but crashed before it could take effect."""
    cfg = _with_overrides(BaselineConfig.load(), tokenizer_path="/tmp/some-tokenizer")
    assert str(cfg.tokenizer_path) == "/tmp/some-tokenizer"


def test_an_unregistered_override_is_rejected() -> None:
    """A typo in a flag name must not silently do nothing."""
    with pytest.raises(CollectorError):
        _with_overrides(BaselineConfig.load(), not_a_real_setting=1)


# ------------------------------------- defect 3: diagnostic-only is a veto on the verdict


def test_diagnostic_only_can_never_pass_baseline_acceptance() -> None:
    """It is a veto, not a label. A diagnostic run cannot become a baseline by accident."""
    passed, problems = _assess_validity(
        cfg=_with_overrides(BaselineConfig.load(), diagnostic_only=True),
        summary=_summary(),
        eligible=[{}] * 4,
        rewards=_rewards(),
    )
    assert passed is False
    assert any("diagnostic_only" in problem for problem in problems)


def test_a_clean_run_passes_acceptance_so_the_veto_is_meaningful() -> None:
    """Without this, the previous test could pass on a function that always fails."""
    cfg = BaselineConfig.load()
    # Enough eligible rollouts and enough distinct rewards to clear every numeric guard.
    eligible = [{}] * cfg.min_valid_rollouts
    rewards = [float(i) / 10 for i in range(cfg.min_distinct_rewards)]
    passed, problems = _assess_validity(
        cfg=cfg,
        summary=_summary(n_reward_eligible=cfg.min_valid_rollouts),
        eligible=eligible,
        rewards=rewards,
    )
    assert passed is True, problems
    assert problems == []


def test_diagnostic_only_never_forgives_an_ineligible_batch() -> None:
    """Diagnostic mode relaxes the overflow guard; it does not make a broken batch valid."""
    passed, problems = _assess_validity(
        cfg=_with_overrides(BaselineConfig.load(), diagnostic_only=True),
        summary=_summary(n_reward_eligible=0),
        eligible=[],
        rewards=[],
    )
    assert passed is False
    assert problems


# -------------------------------- defect 4: context telemetry in the schema's own shape


def test_public_telemetry_uses_the_names_the_schema_accepts() -> None:
    """The schema wants `capacity` and `peak_prompt_tokens`; Probe speaks a different dialect."""
    telemetry = _public_context_telemetry(_probe(), 0)
    assert telemetry is not None
    assert set(telemetry) == {
        "max_model_len",
        "reserved_completion_tokens",
        "safety_margin_tokens",
        "capacity",
        "peak_prompt_tokens",
        "compaction_events",
        "status",
        "context_remaining_tokens",
    }
    assert telemetry["capacity"] == 32768 - 4096 - 512
    assert telemetry["peak_prompt_tokens"] == 1024


def test_public_telemetry_is_scalar_only() -> None:
    """The schema declares scalars here and would reject a nested object."""
    telemetry = _public_context_telemetry(_probe(), 2)
    assert telemetry is not None
    for key, value in telemetry.items():
        assert not isinstance(value, dict | list), f"{key} must be a scalar"
    assert telemetry["compaction_events"] == 2


def test_a_failed_probe_reports_zero_peak_not_null() -> None:
    """A run that failed before the first measurement measured 0 tokens, not "unknown"."""
    telemetry = _public_context_telemetry(_probe(None), 0)
    assert telemetry is not None
    assert telemetry["peak_prompt_tokens"] == 0


def test_no_probe_at_all_yields_no_telemetry_rather_than_a_zeroed_record() -> None:
    """Absent telemetry and telemetry of zero are different facts."""
    assert _public_context_telemetry(None, 0) is None


def test_collected_row_telemetry_validates_against_the_schema() -> None:
    """The whole point of the mapping: the row the collector writes is schema-legal."""
    schema = json.loads((REPO_ROOT / "analysis" / "trace_schema.json").read_text(encoding="utf-8"))
    budget_schema = schema["properties"]["context_budget"]

    telemetry = _public_context_telemetry(_probe(), 1)
    assert telemetry is not None

    allowed = set(budget_schema["properties"])
    assert set(telemetry) <= allowed, f"undeclared keys: {set(telemetry) - allowed}"
    for key in budget_schema.get("required", []):
        assert key in telemetry, f"missing required key {key}"
    assert telemetry["status"] in budget_schema["properties"]["status"]["enum"]


# ------------------------------------- defect 4b: the private sidecar is created 0600


def test_private_log_is_created_with_owner_only_permissions(tmp_path: Path) -> None:
    """A private log holds tool calls and raw transcripts; group and other must never read it."""
    path = tmp_path / "private.jsonl"
    secure_write_private_log(path, [{"rollout_id": "r0"}])
    assert stat.S_IMODE(path.stat().st_mode) == PRIVATE_MODE


def test_resuming_a_private_log_narrows_a_permissive_file(tmp_path: Path) -> None:
    """A file that already exists with wider permissions must be narrowed, not inherited."""
    path = tmp_path / "private.jsonl"
    path.write_text("", encoding="utf-8")
    path.chmod(0o644)

    secure_write_private_log(path, [{"rollout_id": "r0"}])
    assert stat.S_IMODE(path.stat().st_mode) == PRIVATE_MODE


def test_appending_keeps_the_permissions_private(tmp_path: Path) -> None:
    """Append is the resume path, and it must not reopen the file with default permissions."""
    path = tmp_path / "private.jsonl"
    secure_write_private_log(path, [{"rollout_id": "r0"}])
    secure_write_private_log(path, [{"rollout_id": "r1"}], append=True)
    assert stat.S_IMODE(path.stat().st_mode) == PRIVATE_MODE
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_private_log_fails_closed_if_permissions_cannot_be_narrowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private log at the wrong mode is an error, not a warning.

    The writer checks the mode it ended up with and refuses to return if it is not 0600,
    so a filesystem that silently ignores the chmod is caught rather than trusted. The
    check is exercised by making the mode it observes wrong; stubbing `fstat` itself would
    mean rebuilding a `stat_result`, which tests the stub rather than the writer.
    """
    import analysis.trace as trace_module

    real_stat = trace_module.os.stat

    class _WideStat:
        """A stat result that reports group/other read, whatever is on disk."""

        def __init__(self, inner: Any) -> None:
            self._inner = inner
            self.st_mode = (inner.st_mode & ~0o777) | 0o664

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    monkeypatch.setattr(trace_module.os, "fstat", lambda fd: _WideStat(real_stat(fd)))
    with pytest.raises(TraceError):
        secure_write_private_log(tmp_path / "private.jsonl", [{"rollout_id": "r0"}])


# ------------------------------------------- defect 5: the manifest reports the run


def _manifest(**kwargs: Any) -> dict[str, Any]:
    from scripts.run_manifest import build_manifest

    base = {
        "arm": "pilot",
        "run_id": "phase6f-test",
        "split": "train",
        "seed": 0,
        "input_checkpoint": None,
        "output_checkpoint": None,
        "note": None,
    }
    base.update(kwargs)
    return build_manifest(**base)


def test_manifest_effective_block_reports_what_the_merged_document_says() -> None:
    """`effective` must equal the merged overlay value, not the protocol's, and say so.

    This test previously asserted `effective["rollouts_per_task"] == 4` with the reasoning
    "Phase 6E's manifest said 8 while the protocol says 4". That reasoning was wrong twice
    over. First, 4 was the *protocol* value and the pilot overlay deliberately raises it to
    8, so the merge is supposed to yield 8. Second, the number only came out as 4 because
    `_deep_merge` was writing nested overlay keys onto the root document, which left
    `rollouts` holding the protocol's untouched copy. The assertion was passing on the bug.

    The property that actually matters is the one stated in the test name: `effective` is
    the merged value with its provenance, and it is never silently the protocol default.
    """
    from scripts.run_manifest import load_pilot, load_protocol

    protocol, pilot = load_protocol(), load_pilot()
    effective = _manifest()["effective"]

    assert effective["rollouts_per_task"] == pilot["rollouts"]["rollouts_per_task"]
    assert effective["rollouts_per_task"] != protocol["rollouts"]["rollouts_per_task"]
    assert "pilot_rollouts_1p5b.yaml" in effective["provenance"]["rollouts.rollouts_per_task"]


def test_manifest_tasks_stay_null_because_neither_file_names_a_task() -> None:
    """A null `tasks` is a visible gap, not a default - and it must stay visible."""
    manifest = _manifest()
    assert manifest["effective"]["tasks"] is None
    assert manifest["pilot"]["values"]["tasks"] is None


def test_manifest_records_three_hashes_and_per_field_provenance() -> None:
    """Three documents feed a run, so three hashes; and each value names its source."""
    hashes = _manifest()["effective"]["hashes"]
    assert set(hashes) == {"protocol", "overlay", "merged"}
    for name in ("protocol", "overlay"):
        assert hashes[name]["sha256"], f"{name} hash must be present"
    assert len(hashes["merged"]["sha256"]) == 64
    assert hashes["merged"]["bytes"] > 0

    provenance = _manifest()["effective"]["provenance"]
    assert provenance, "provenance must not be empty"
    assert "configs/training/pilot_rollouts_1p5b.yaml" in provenance["rollouts.rollouts_per_task"]


def test_manifest_still_names_the_overlay_so_its_defaults_cannot_masquerade() -> None:
    """The `pilot` block keeps the file's own values whatever the arm resolved to."""
    manifest = _manifest()
    assert manifest["pilot"]["values"]["rollouts_per_task"] == 8
    assert manifest["pilot"]["values"]["tasks"] is None
    assert manifest["pilot"]["applies"] is True
    # `effective` is the merged document, and the merge is not the identity.
    assert manifest["effective"]["hashes"]["merged"]["sha256"]
    assert manifest["effective"]["hashes"]["protocol"]["sha256"] != manifest["effective"]["hashes"]["merged"]["sha256"]


def test_manifest_merged_hash_moves_when_the_run_settings_change() -> None:
    """A fingerprint that does not move is not a fingerprint."""
    from scripts.run_manifest import effective_settings, load_pilot, load_protocol

    protocol, pilot = load_protocol(), load_pilot()
    first = effective_settings(protocol, pilot, arm="pilot")
    second = effective_settings(protocol, pilot, arm="pilot", overrides={"measured_tasks": 1})
    assert first["hashes"]["merged"]["sha256"] != second["hashes"]["merged"]["sha256"]
    assert first["hashes"]["protocol"] == second["hashes"]["protocol"]


def test_nested_overlay_keys_land_inside_their_own_section() -> None:
    """A nested overlay key must merge into its section, not onto the root document.

    The first `_deep_merge` recursed into the overlay's nested dicts but wrote each leaf
    onto the ROOT copy, so `hardware.concurrency: 1` became a top-level `concurrency` and
    `hardware.concurrency` stayed absent. Scalars like `rollouts_per_task` were unaffected,
    which is exactly why the other tests in this file passed: they only inspected the
    values the bug happened to get right.
    """
    from scripts.run_manifest import _deep_merge, load_pilot, load_protocol

    protocol, pilot = load_protocol(), load_pilot()
    merged, provenance = _deep_merge(protocol, pilot)

    # No overlay key may appear at the root unless the overlay declares it at the root.
    root_only = {"meta", "purpose", "observability"}
    assert set(merged) - set(protocol) <= root_only
    assert "concurrency" not in merged, "hardware.concurrency leaked onto the root document"
    assert "shutdown_required" not in merged, "hardware.shutdown_required leaked onto the root document"

    # ...and the values are where the schema says they are.
    assert merged["hardware"]["concurrency"] == pilot["hardware"]["concurrency"]
    assert merged["hardware"]["shutdown_required"] is True
    assert provenance["hardware.concurrency"] == "overlay"


def test_effective_block_reports_concurrency_from_the_hardware_section() -> None:
    """`concurrency` is the value a reader needs to reproduce the run, so it must resolve."""
    effective = _manifest()["effective"]
    assert effective["concurrency"] is not None
    assert isinstance(effective["concurrency"], int)
    assert effective["provenance"]["hardware.concurrency"] == "overlay"


def test_merge_of_an_empty_overlay_is_the_identity() -> None:
    """Keys the overlay never mentions must survive the merge untouched."""
    from scripts.run_manifest import _deep_merge, load_protocol

    protocol = load_protocol()
    merged, provenance = _deep_merge(protocol, {})
    assert merged == protocol
    assert provenance == {}


# --------------------------------------------------- standing prohibitions, re-asserted


def test_heldout_split_remains_prohibited_for_the_pilot_arm() -> None:
    """Phase 5's seal outranks any convenience flag; re-asserted because Phase 6F edited the CLI."""
    from scripts.run_manifest import NON_UPDATING_ARMS

    assert "pilot" in NON_UPDATING_ARMS


def test_reward_v2_is_offline_only_so_no_run_may_compute_it_inline() -> None:
    """v2 was built as an analysis-time reward; computing it inline would be a protocol change."""
    protocol = yaml.safe_load((REPO_ROOT / "configs" / "training" / "protocol_1p5b.yaml").read_text(encoding="utf-8"))
    reward = protocol.get("reward") or {}
    assert reward, "the protocol must state its reward configuration"
    assert reward.get("offline_only") is not False


# --- the `executed` block ---------------------------------------------------------------
#
# The manifest hashes the *training* pair (`protocol_1p5b.yaml` + `pilot_rollouts_1p5b.yaml`),
# but a diagnostic smoke does not run that pair: it runs a materialised eval config, and
# `collect_baseline_rollouts.py` resolves no `extends:`, so only the materialised file is
# ever loaded. These tests pin the `executed` block that closes that gap.


def test_executed_block_reads_the_settings_the_collector_will_actually_load() -> None:
    """A materialised config must report the run-level facts, not a null task list."""
    from scripts.run_manifest import executed_config

    rel = "results/raw/phase6f_unit_probe/merged_config.yaml"
    target = REPO_ROOT / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    materialised = {
        "model": {"name": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "revision": "abc123", "adapter": None},
        "sampling": {"temperature": 1.0, "top_p": 1.0, "seed": 42, "max_tokens_per_turn": 1024},
        "episode": {
            "max_turns": "from_task_yaml",
            "max_tool_calls": "from_task_yaml",
            "timeout_s_per_rollout": 900,
        },
        "run": {
            "task_subset": ["bugfix_offbyone_pagination"],
            "rollouts_per_task": 4,
            "min_valid_rollouts": 4,
            "min_distinct_rewards": 2,
            "diagnostic_only": True,
        },
        "output": {
            "run_id": "phase6f_unit_probe",
            # The collector opens these three by name (`output_path("rollouts")` etc.), so a
            # config without them is not runnable and `materialised` says so.
            "rollouts": "results/raw/phase6f_unit_probe/rollouts.jsonl",
            "private": "results/raw/phase6f_unit_probe/rollouts_private.jsonl",
            "failures": "results/raw/phase6f_unit_probe/failures.jsonl",
        },
        "tasks": {"split": "train", "heldout_enabled": False},
        "reward": {"live": "reward_v1", "analysed": "reward_v2", "reward_v2_computed_inline": False},
        "server": {"cuda_visible_devices": "0", "served_model_name": "Qwen/Qwen2.5-Coder-1.5B-Instruct"},
        "context": {
            "max_model_len": 8192,
            "reserved_completion_tokens": 1024,
            "safety_margin_tokens": 64,
        },
    }
    try:
        target.write_text(yaml.safe_dump(materialised, sort_keys=False), encoding="utf-8")
        block = executed_config(rel)
    finally:
        target.unlink(missing_ok=True)
        if target.parent.is_dir() and not any(target.parent.iterdir()):
            target.parent.rmdir()

    assert block["present"] is True
    assert block["materialised"] is True
    assert block["settings"]["tasks"] == ["bugfix_offbyone_pagination"]
    assert block["settings"]["rollouts_per_task"] == 4
    assert block["settings"]["diagnostic_only"] is True
    assert block["settings"]["heldout_enabled"] is False
    assert len(block["sha256"]) == 64


def test_executed_block_refuses_an_overlay_that_nothing_ever_expands() -> None:
    """An overlay is not runnable: `extends:` is inert, so its absent keys would raise."""
    from scripts.run_manifest import executed_config

    block = executed_config("configs/eval/phase6g_diagnostic_smoke.yaml")

    assert block["present"] is True, "the overlay exists; it is simply not materialised"
    assert block["materialised"] is False
    assert block["error"] and "extends" in block["error"]
    assert block["settings"]["model_name"] is None, "no base was merged, so model.name is genuinely absent"


def test_executed_block_keeps_a_missing_config_loud() -> None:
    """A path that does not exist must not be reported as an empty-but-fine executed config."""
    from scripts.run_manifest import executed_config

    block = executed_config("results/raw/does_not_exist_6f/merged_config.yaml")

    assert block["present"] is False
    assert block["materialised"] is False
    assert block["error"]
    assert block["sha256"] is None
