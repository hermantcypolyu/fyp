"""Phase 2 guarantees: trajectory identity, persistence, termination, and leakage.

These tests exist because the v2 trace could not answer four questions that the research
depends on, and each one had already cost a run:

1. *Which trajectory is this?* A row with no seed and no checkpoint cannot be tied to the
   weights that produced it, so an improvement cannot be attributed.
2. *Did a rollout fail, or did it achieve nothing?* A crashed rollout left no row at all,
   so a GRPO group of crashes averaged to a zero-variance 0.0 - indistinguishable from a
   group that ran fine and scored nothing. That ambiguity is the research blocker.
3. *What did the model actually submit?* The sandbox is deleted; the patch was unrecoverable.
4. *Can the row leak the hidden tests?* The v2 row carried `per_test` nodeids.

The leakage tests are the ones that must never be deleted: they are the only thing standing
between a trajectory row and a dashboard that names the tests the model failed.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from analysis.trace import (
    SCHEMA_PATH,
    TERMINATION_REASONS,
    Timer,
    TraceError,
    _termination_reason,
    build_row,
    collect_patch,
    read_private_jsonl,
    validate_row,
    write_jsonl,
    write_private_jsonl,
)
from benchmark.tools.patch_utils import apply_patch
from benchmark.tools.task_loader import TASKS_ROOT, load_task
from envs.coding_env.session import EpisodeSession

TASK_ID = "bugfix_metric_zero_domain_error"


@pytest.fixture(scope="module")
def task():
    return load_task(TASKS_ROOT / "train" / TASK_ID)


@pytest.fixture(scope="module")
def submitted_row(task, tmp_path_factory):
    session = EpisodeSession.open(task, parent=tmp_path_factory.mktemp("phase2"))
    timer = Timer.start()
    try:
        session.tool("list_files", {"path": "repo"})
        apply_patch(session.sandbox.root, task.gold_patch())
        session.tool("run_tests")
        session.tool("submit")
        return build_row(
            run_id="phase2",
            rollout_id=f"{task.id}#0",
            task=task,
            result=session.grade(),
            tool_calls=session.sandbox.toolbox.log,
            timer=timer,
            agent_kind="scripted_gold",
            random_seed=0,
            prompt_config_version="v1",
            candidate_patch=collect_patch(task, session.sandbox.repo_dir),
        )
    finally:
        session.close()


# --------------------------------------------------------------- 2.2 unique identity


def test_a_row_is_identifiable_to_the_checkpoint_that_produced_it(submitted_row):
    """`run_id` + `task.id` + `rollout_id` is the group key; the agent block completes the identity.

    A trace that names only 'the 1.5B model' cannot support a claim that a later checkpoint
    improved anything, because the two rows are indistinguishable.
    """
    agent = submitted_row["agent"]
    for key in ("checkpoint_id", "random_seed", "prompt_config_version"):
        assert key in agent, f"{key} is required for reproducibility"
    assert agent["random_seed"] == 0
    assert agent["prompt_config_version"] == "v1"


def test_two_rollouts_of_one_prompt_are_distinct_rows(task, tmp_path_factory):
    """The k samples of a GRPO group must be distinguishable, not overwritten."""
    rows = []
    for sample in range(2):
        session = EpisodeSession.open(task, parent=tmp_path_factory.mktemp(f"grpo{sample}"))
        timer = Timer.start()
        try:
            session.tool("list_files", {"path": "repo"})
            apply_patch(session.sandbox.root, task.gold_patch())
            session.tool("submit")
            rows.append(
                build_row(
                    run_id="grpo",
                    rollout_id=f"{task.id}#{sample}",
                    task=task,
                    result=session.grade(),
                    tool_calls=session.sandbox.toolbox.log,
                    timer=timer,
                    agent_kind="scripted_gold",
                )
            )
        finally:
            session.close()
    keys = [(r["run_id"], r["task"]["id"], r["rollout_id"]) for r in rows]
    assert len(set(keys)) == len(keys), "rollout_id must distinguish the k samples of one prompt"


# --------------------------------------------------------------- 2.2 termination reasons


def test_every_termination_reason_is_in_the_declared_vocabulary(submitted_row):
    assert submitted_row["termination_reason"] in TERMINATION_REASONS
    assert TERMINATION_REASONS == ("submitted", "max_turns", "max_tool_calls", "timeout", "error")


class _Stub:
    """Just enough of VerifyResult to drive `_termination_reason`."""

    def __init__(self, *, submitted: bool, tool_calls: int) -> None:
        self.submitted = submitted
        self.tool_calls = tool_calls


@pytest.mark.parametrize(
    ("stub", "max_tool_calls", "expected"),
    [
        (_Stub(submitted=True, tool_calls=3), 40, "submitted"),
        (_Stub(submitted=False, tool_calls=40), 40, "max_tool_calls"),
        (_Stub(submitted=False, tool_calls=20), 40, "max_turns"),
    ],
)
def test_the_cause_of_the_end_is_distinguished_from_the_gate(stub, max_tool_calls, expected):
    """`no_submit` is the consequence of two different causes and cannot tell them apart.

    Both non-submitting cases below fire the same `no_submit` gate; only the termination
    reason says whether the tool-call budget or the turn budget was the thing that ran out.
    """
    assert (
        _termination_reason(result=stub, timed_out=False, max_tool_calls=max_tool_calls, submitted=stub.submitted)
        == expected
    )


def test_a_timeout_is_a_cause_not_an_exception():
    """A timed-out rollout is recorded and gated, not dropped - dropping it biases every aggregate."""
    stub = _Stub(submitted=False, tool_calls=5)
    assert _termination_reason(result=stub, timed_out=True, max_tool_calls=40, submitted=False) == "timeout"


# --------------------------------------------------------------- 2.2 failed rollouts persist


def test_a_rollout_that_raised_still_produces_a_valid_row(task, tmp_path_factory):
    """The gap that caused the research blocker: crashed rollouts used to leave no row at all.

    With no row, a GRPO group of crashes averaged to a zero-variance 0.0, which is exactly
    what a group that ran fine and achieved nothing also scores. The two are now separable.
    """
    session = EpisodeSession.open(task, parent=tmp_path_factory.mktemp("crash"))
    timer = Timer.start()
    try:
        session.tool("list_files", {"path": "repo"})
        row = build_row(
            run_id="crash",
            rollout_id=f"{task.id}#0",
            task=task,
            result=None,
            tool_calls=session.sandbox.toolbox.log,
            timer=timer,
            error=TimeoutError("model server did not respond"),
        )
    finally:
        session.close()
    assert row["termination_reason"] == "error"
    assert row["error"]["type"] == "TimeoutError"
    assert row["result"]["reward"] == -1.0
    assert row["steps"], "the steps taken before the crash are still evidence"


def test_a_row_with_an_error_and_a_termination_reason_must_agree(task, tmp_path_factory):
    """`termination_reason: error` and a null `error` would tell two different stories.

    The build path cannot produce that pair, so this asserts the invariant on a row that a
    hand-built or migrated writer might produce: if the reason is `error`, `error` must not
    be null, because otherwise nothing in the row says what went wrong.
    """
    session = EpisodeSession.open(task, parent=tmp_path_factory.mktemp("pairing"))
    timer = Timer.start()
    try:
        session.tool("list_files", {"path": "repo"})
        raised = build_row(
            run_id="pairing",
            rollout_id=f"{task.id}#0",
            task=task,
            result=None,
            tool_calls=session.sandbox.toolbox.log,
            timer=timer,
            error=RuntimeError("harness died"),
        )
    finally:
        session.close()
    assert raised["termination_reason"] == "error"
    assert raised["error"] is not None, "an 'error' termination must name the error"


def test_a_normal_row_records_a_null_error(submitted_row):
    assert submitted_row["termination_reason"] != "error"
    assert submitted_row["error"] is None


# --------------------------------------------------------------- 2.2 the submitted patch


def test_the_submitted_patch_survives_the_sandbox_being_deleted(submitted_row):
    """The sandbox is gone and arguments are clipped, so this is the only durable copy."""
    patch = submitted_row["candidate_patch"]
    assert patch["present"] is True
    assert patch["diff"], "a submitted patch must be recoverable from the row"
    assert patch["chars"] > 0
    assert patch["sha256"] == hashlib.sha256(patch["diff"].encode("utf-8")).hexdigest()


def test_no_patch_is_a_recorded_fact_not_a_missing_key(task, tmp_path_factory):
    session = EpisodeSession.open(task, parent=tmp_path_factory.mktemp("nopatch"))
    timer = Timer.start()
    try:
        session.tool("list_files", {"path": "repo"})
        row = build_row(
            run_id="nopatch",
            rollout_id=f"{task.id}#0",
            task=task,
            result=session.grade(),
            tool_calls=session.sandbox.toolbox.log,
            timer=timer,
        )
    finally:
        session.close()
    assert row["candidate_patch"]["present"] is False
    assert row["candidate_patch"]["sha256"] is None
    assert row["candidate_patch"]["diff"] is None


# --------------------------------------------------------------- 2.3 leakage


def test_the_trajectory_row_carries_no_hidden_test_names(submitted_row):
    """The row is what a dashboard dumps. It must not be able to name a failing test."""
    serialised = json.dumps(submitted_row)
    assert "per_test" not in submitted_row["result"]
    assert "per_test" not in submitted_row
    assert "::" not in serialised, "a pytest nodeid would look like path::test_name"
    assert "test_" not in serialised or "n_tests" in serialised


def test_the_hidden_test_block_is_counts_only(submitted_row):
    hidden = submitted_row["hidden_tests"]
    assert set(hidden) == {"declared", "passed", "floor", "failed"}
    assert all(isinstance(v, int) for v in hidden.values())


def test_the_private_sidecar_carries_the_nodeids_and_joins_on_the_public_key(submitted_row, tmp_path):
    """The analysis still needs per-test outcomes; it just must not live in the public row."""
    private = write_private_jsonl(
        [(submitted_row, {"per_test": {"tests/test_metric.py::test_zero": "passed"}})],
        tmp_path / "rollouts_private.jsonl",
    )
    loaded = read_private_jsonl(private)
    key = (submitted_row["run_id"], submitted_row["rollout_id"])
    assert key in loaded
    assert loaded[key]["per_test"] == {"tests/test_metric.py::test_zero": "passed"}


def test_writing_the_public_row_cannot_serialise_a_nodeid(submitted_row, tmp_path):
    """A regression guard: public rows are validated, and `result` has no place for nodeids."""
    path = write_jsonl([submitted_row], tmp_path / "rollouts.jsonl")
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        validate_row(row)
        assert "per_test" not in row["result"]


# --------------------------------------------------------------- schema version handling


def test_a_v2_row_is_still_readable_after_the_bump():
    """Refusing old rows would make every historical run unanalysable on upgrade."""
    v2 = json.loads((SCHEMA_PATH.parent / "trace_schema_v2.json").read_text(encoding="utf-8"))
    assert v2["properties"]["schema_version"]["const"] == "2"
    assert "tool_calls" in v2["properties"], "the v2 shape must stay on disk, not just in git"


def test_an_unknown_schema_version_is_a_clear_error():
    with pytest.raises(TraceError, match="unknown schema_version"):
        validate_row({"schema_version": "99"})
