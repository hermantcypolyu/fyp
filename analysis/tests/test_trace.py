"""Rollout traces and pinned versions.

The trace is the only durable record of a rollout, so the tests that matter here are the
*rejections*: a row that is silently accepted with a missing field, an unknown gate or a
stray key is a dataset that looks fine until the moment someone tries to analyse it.

`test_a_real_episode_produces_a_valid_row` is the one that would catch real drift - it runs
an actual episode and validates what comes out, rather than validating a row written by
hand to match the schema it is being checked against.
"""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from analysis import versions
from analysis.trace import (
    MAX_FIELD_CHARS,
    SCHEMA_PATH,
    Timer,
    TraceError,
    build_row,
    read_jsonl,
    validate_row,
    write_jsonl,
)
from benchmark.tools.patch_utils import apply_patch
from benchmark.tools.task_loader import TASKS_ROOT, load_task
from envs.coding_env.session import EpisodeSession

TASK_ID = "bugfix_metric_zero_domain_error"
DOCS = SCHEMA_PATH.resolve().parents[1] / "docs"


@pytest.fixture(scope="module")
def task():
    return load_task(TASKS_ROOT / "train" / TASK_ID)


@pytest.fixture(scope="module")
def real_row(task, tmp_path_factory):
    """One genuine episode, graded, turned into a trace row."""
    session = EpisodeSession.open(task, parent=tmp_path_factory.mktemp("trace"))
    timer = Timer.start()
    try:
        session.tool("list_files", {"path": "repo"})
        apply_patch(session.sandbox.root, task.gold_patch())
        session.tool("run_tests")
        session.tool("submit")
        return build_row(
            run_id="test_run",
            rollout_id=f"{task.id}#0",
            task=task,
            result=session.grade(),
            tool_calls=session.sandbox.toolbox.log,
            timer=timer,
            agent_kind="scripted_gold",
        )
    finally:
        session.close()


# --------------------------------------------------------------------------- the schema


def test_schema_is_itself_a_valid_json_schema():
    validate_row  # noqa: B018 - importing the module already ran check_schema
    assert json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))["$schema"].endswith("2020-12/schema")


# --------------------------------------------------------------------------- a real row


@pytest.mark.slow
def test_a_real_episode_produces_a_valid_row(real_row):
    """build_row validates on the way out, so reaching here is already the assertion."""
    assert real_row["result"]["reward"] == 1.0
    assert real_row["result"]["tests_passed"] == real_row["task"]["n_tests_declared"]


@pytest.mark.slow
def test_every_field_is_populated_not_placeholder(real_row):
    assert real_row["task"]["provenance_source"] in {"real_repo", "invented"}
    assert real_row["budgets"]["tool_calls_used"] == len(real_row["tool_calls"])
    assert real_row["duration_s"] > 0
    assert real_row["result"]["per_test"], "per-test detail is what makes a failure taxonomy possible"
    assert real_row["versions"]["repo_commit"], "a trace that cannot name its code is not reproducible"
    assert set(real_row["limits_enforced"]) >= {"wall_clock_s", "network"}


@pytest.mark.slow
def test_tool_transcript_is_ordered_and_complete(real_row):
    assert [c["index"] for c in real_row["tool_calls"]] == list(range(len(real_row["tool_calls"])))
    assert [c["name"] for c in real_row["tool_calls"]] == ["list_files", "run_tests", "submit"]


# --------------------------------------------------------------------------- rejections


@pytest.mark.slow
@pytest.mark.parametrize("field", ["run_id", "task", "result", "versions", "limits_enforced"])
def test_a_missing_top_level_field_is_rejected(real_row, field):
    row = deepcopy(real_row)
    del row[field]
    with pytest.raises(TraceError, match=field):
        validate_row(row)


@pytest.mark.slow
def test_an_unknown_gate_name_is_rejected(real_row):
    """Gates are the reward's hard zeros; a typo'd one would silently never be counted."""
    row = deepcopy(real_row)
    row["result"]["gates"] = ["tests_modifed"]
    with pytest.raises(TraceError, match="gates"):
        validate_row(row)


@pytest.mark.slow
def test_a_stray_key_is_rejected(real_row):
    row = deepcopy(real_row)
    row["extra_thought"] = "n/a"
    with pytest.raises(TraceError):
        validate_row(row)


@pytest.mark.slow
def test_pass_fraction_outside_zero_to_one_is_rejected(real_row):
    row = deepcopy(real_row)
    row["result"]["pass_fraction"] = 1.5
    with pytest.raises(TraceError, match="pass_fraction"):
        validate_row(row)


# --------------------------------------------------------------------------- truncation


@pytest.mark.slow
def test_long_arguments_are_clipped_but_the_length_is_kept(task, tmp_path):
    session = EpisodeSession.open(task, parent=tmp_path)
    timer = Timer.start()
    try:
        session.tool("write_file", {"path": "repo/humanize/big.py", "content": "# pad\n" * 5000})
        session.tool("submit")
        row = build_row(
            run_id="r",
            rollout_id="r#0",
            task=task,
            result=session.grade(),
            tool_calls=session.sandbox.toolbox.log,
            timer=timer,
        )
    finally:
        session.close()
    call = row["tool_calls"][0]
    assert len(call["arguments"]["content"]) < 30_000
    assert "chars]" in call["arguments"]["content"]
    assert call["output_chars"] >= len(call["output"]) - MAX_FIELD_CHARS


# --------------------------------------------------------------------------- round trip


@pytest.mark.slow
def test_jsonl_round_trip(real_row, tmp_path):
    path = write_jsonl([real_row, real_row], tmp_path / "rollouts.jsonl")
    rows = list(read_jsonl(path))
    assert len(rows) == 2
    assert rows[0] == real_row


def test_write_revalidates_so_a_hand_built_row_cannot_slip_through(tmp_path):
    with pytest.raises(TraceError):
        write_jsonl([{"schema_version": "1"}], tmp_path / "bad.jsonl")


# --------------------------------------------------------------------------- versions


def test_pins_are_full_length_sha1s():
    for name, spec in versions.PINNED.items():
        assert len(spec["commit"]) == 40, name
        int(spec["commit"], 16)  # raises if not hex


def test_resolve_reports_the_actual_checkout_and_whether_it_matches():
    pin = versions.resolve("nemo_gym")
    assert pin.pinned == versions.PINNED["nemo_gym"]["commit"]
    if pin.actual is None:
        pytest.skip("_vendor/nemo-gym not present on this machine")
    assert pin.matches is (pin.actual == pin.pinned)


def test_a_missing_clone_is_not_an_error(tmp_path):
    """The clones live outside the repo and are gitignored; absence is normal."""
    pin = versions.resolve("nemo_rl", vendor_root=tmp_path)
    assert pin.actual is None and pin.matches is False


def test_the_two_stacks_disagree_about_python_and_that_is_recorded():
    """NeMo RL needs 3.13, NeMo Gym runs on 3.12. Losing this fact costs a day on setup."""
    assert versions.PINNED["nemo_gym"]["requires_python"] == ">=3.12"
    assert versions.PINNED["nemo_rl"]["requires_python"] == ">=3.13.13,<3.14"


@pytest.mark.parametrize("doc", ["pinned_versions.md", "SETUP_NEW_MACHINE.md"])
def test_docs_quote_the_same_commits_as_the_code(doc):
    """Guards the one thing a human will copy-paste from: the clone commands."""
    text = (DOCS / doc).read_text(encoding="utf-8")
    for name, spec in versions.PINNED.items():
        assert spec["commit"] in text, f"{doc} does not name the pinned {name} commit"
