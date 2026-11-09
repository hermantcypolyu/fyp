"""Part C: CPU-only tests for the Phase 7 input-manifest generator.

Exercises the manifest against synthetic rollout files. The real Phase 6M file is
covered by the run-time verification, not here: a test that read `results/` would
fail on a fresh clone, where `results/` is legitimately absent because it is
gitignored.

The tests focus on the two things the manifest is for: fingerprinting the exact
rows, and refusing to describe anything that is not the Phase 7 group.
"""

from __future__ import annotations

import json

import pytest

from scripts.build_phase7_input_manifest import (
    ABSENT_ANALYSIS_PATH,
    build_manifest,
    canonical_row_sha256,
    normalise_for_guard,
    reward_of,
    rollout_index_of,
    seed_of,
    task_id_of,
)

TASK_ID = "bugfix_offbyone_pagination"
REWARDS = (0.0, -1.0, -1.0, -1.0)


def make_trace_row(index: int, reward: float) -> dict:
    """A row shaped like the recorded Phase 6M trace: identity fields are nested."""
    return {
        "rollout_id": f"phase6m_test__{TASK_ID}__r{index:02d}",
        "run_id": "phase6m_test",
        "schema_version": 3,
        "task": {"id": TASK_ID, "split": "train"},
        "agent": {"rollout_index": index, "sampling_seed": 1000 + index},
        "result": {"reward": reward, "submitted": reward == 0.0},
        "termination_reason": "submitted" if reward == 0.0 else "max_turns",
        "error": None,
    }


def write_trace(tmp_path, rows: list[dict]):
    path = tmp_path / "rollouts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def make_rows(rewards=REWARDS) -> list[dict]:
    return [make_trace_row(index, reward) for index, reward in enumerate(rewards)]


def test_nested_fields_are_read_from_the_trace_shape():
    row = make_trace_row(2, -1.0)

    assert rollout_index_of(row) == 2
    assert reward_of(row) == -1.0
    assert seed_of(row) == 1002
    assert task_id_of(row) == TASK_ID


def test_flat_fallback_shape_also_works():
    row = {"task_id": TASK_ID, "rollout_index": 1, "reward": 0.5, "sampling_seed": 7}

    assert rollout_index_of(row) == 1
    assert reward_of(row) == 0.5
    assert seed_of(row) == 7
    assert task_id_of(row) == TASK_ID


def test_normalise_projects_nested_rows_onto_the_guard_shape():
    flat = normalise_for_guard(make_rows())

    assert [row["rollout_index"] for row in flat] == [0, 1, 2, 3]
    assert [row["reward"] for row in flat] == list(REWARDS)
    assert all(row["task_id"] == TASK_ID for row in flat)
    assert all(row["eligible_for_training"] is True for row in flat)


def test_row_digest_is_stable_across_key_order():
    row = make_trace_row(0, 0.0)
    reordered = dict(reversed(list(row.items())))

    assert canonical_row_sha256(row) == canonical_row_sha256(reordered)


def test_row_digest_changes_when_a_reward_changes():
    before = canonical_row_sha256(make_trace_row(0, 0.0))
    after = canonical_row_sha256(make_trace_row(0, -1.0))

    assert before != after


def test_manifest_records_the_canonical_group(tmp_path):
    manifest = build_manifest(write_trace(tmp_path, make_rows()))

    group = manifest["group"]

    assert group["rewards"] == [0.0, -1.0, -1.0, -1.0]
    assert group["rewards_match_canonical_sequence"] is True
    assert group["rollout_count"] == 4
    assert group["distinct_reward_values"] == 2
    assert group["reward_mean"] == -0.75
    assert group["task_id"] == TASK_ID
    assert [row["rollout_index"] for row in group["rows"]] == [0, 1, 2, 3]
    assert [row["sampling_seed"] for row in group["rows"]] == [1000, 1001, 1002, 1003]


def test_manifest_records_advantages_consistent_with_repository_math(tmp_path):
    manifest = build_manifest(write_trace(tmp_path, make_rows()))
    advantages = manifest["group"]["advantages"]

    assert advantages[0] == pytest.approx(1.732046807578115, abs=1e-12)
    assert advantages[1] == pytest.approx(-0.5773489358593717, abs=1e-12)
    assert manifest["group"]["reward_std_population"] == pytest.approx(0.4330127018922193)


def test_manifest_pins_model_and_execution_constraints(tmp_path):
    manifest = build_manifest(write_trace(tmp_path, make_rows()))

    assert manifest["model"]["name"] == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert manifest["model"]["revision"] == "488639f1ff808d1d3d0ba301aef8c11461451ec5"
    assert manifest["model"]["seven_b_prohibited"] is True
    assert manifest["model"]["one_point_five_b_prohibited"] is True

    execution = manifest["execution"]
    assert execution["expected_optimizer_steps"] == 1
    assert execution["iterations"] == 1
    assert execution["lora_only"] is True
    assert execution["existing_adapter_prohibited"] is True
    assert execution["rollout_generation_disabled"] is True
    assert execution["heldout_evaluation_disabled"] is True
    assert execution["kl_status"] == "NOT_APPLICABLE_CURRENT_IMPLEMENTATION"


def test_manifest_is_labelled_newly_generated(tmp_path):
    manifest = build_manifest(write_trace(tmp_path, make_rows()))

    assert manifest["newly_generated"] is True
    assert manifest["generated_utc"]


def test_manifest_records_the_absent_analysis_json_as_absent(tmp_path):
    manifest = build_manifest(write_trace(tmp_path, make_rows()))

    (record,) = manifest["absent_source_artifacts"]

    assert record["path"] == ABSENT_ANALYSIS_PATH
    # Not reconstructed, and not claimed to exist.
    assert record["present"] is False
    assert "NOT reconstructed" in record["note"]


def test_manifest_records_rollouts_file_digest_and_line_count(tmp_path):
    path = write_trace(tmp_path, make_rows())
    manifest = build_manifest(path)

    assert manifest["source_run"]["rollouts_line_count"] == 4
    assert len(manifest["source_run"]["rollouts_sha256"]) == 64


def test_manifest_refuses_five_rows(tmp_path):
    rows = make_rows() + [make_trace_row(4, -1.0)]

    with pytest.raises(SystemExit, match="non-Phase-7 group"):
        build_manifest(write_trace(tmp_path, rows))


def test_manifest_refuses_three_rows(tmp_path):
    with pytest.raises(SystemExit, match="non-Phase-7 group"):
        build_manifest(write_trace(tmp_path, make_rows()[:3]))


def test_manifest_refuses_permuted_rewards(tmp_path):
    with pytest.raises(SystemExit, match="non-Phase-7 group"):
        build_manifest(write_trace(tmp_path, make_rows((-1.0, 0.0, -1.0, -1.0))))


def test_manifest_refuses_uniform_rewards(tmp_path):
    with pytest.raises(SystemExit, match="non-Phase-7 group"):
        build_manifest(write_trace(tmp_path, make_rows((-1.0, -1.0, -1.0, -1.0))))


def test_manifest_refuses_mixed_tasks(tmp_path):
    rows = make_rows()
    rows[2]["task"] = {"id": "bugfix_filesize_unit_rollover", "split": "train"}

    with pytest.raises(SystemExit, match="non-Phase-7 group"):
        build_manifest(write_trace(tmp_path, rows))


def test_manifest_refuses_a_row_with_an_error(tmp_path):
    """A rollout carrying an infrastructure error is not a training candidate."""
    rows = make_rows()
    rows[1]["error"] = "vllm timeout"

    with pytest.raises(SystemExit, match="non-Phase-7 group"):
        build_manifest(write_trace(tmp_path, rows))
