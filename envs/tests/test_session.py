"""Episode session state - the whole resources server minus the routing.

`app.py` cannot be imported on Windows (`nemo_gym.server_utils` does `import resource`), so
this file is where the server's behaviour is actually pinned down. If these pass, the only
untested thing left in the HTTP layer is a list of FastAPI decorators.
"""

from __future__ import annotations

import pytest

from benchmark.tools.task_loader import TASKS_ROOT, TaskError, iter_tasks, load_task
from envs.coding_env.session import EpisodeSession, SessionStore, episode_metadata, find_task
from envs.coding_env.tools import GATE_TESTS_MODIFIED
from envs.coding_env.verifier import GATE_NO_SUBMIT

TASK_ID = "bugfix_metric_zero_domain_error"


@pytest.fixture(scope="module")
def task():
    return load_task(TASKS_ROOT / "train" / TASK_ID)


@pytest.fixture
def session(task, tmp_path):
    ep = EpisodeSession.open(task, parent=tmp_path)
    yield ep
    ep.close()


# --------------------------------------------------------------------------- task lookup


def test_find_task_resolves_by_id(task):
    assert find_task(TASK_ID).id == task.id


def test_find_task_names_the_id_it_could_not_find():
    with pytest.raises(TaskError, match="nonexistent_task"):
        find_task("nonexistent_task")


# --------------------------------------------------------------------------- budget wiring


def test_metadata_carries_the_task_s_own_budgets(task):
    constraints = task.data["constraints"]
    meta = episode_metadata(task)
    assert meta["max_turns"] == str(constraints["max_turns"])
    assert meta["max_tool_calls"] == str(constraints["max_tool_calls"])
    assert meta["task_id"] == task.id


def test_budgets_really_do_vary_across_the_corpus():
    """If they did not, per-task wiring would be untested ceremony."""
    turns = {episode_metadata(t)["max_turns"] for t in iter_tasks(("train",))}
    assert len(turns) > 1, f"expected varying max_turns, got {turns}"


def test_every_task_s_metadata_matches_its_yaml():
    for t in iter_tasks(("train",)):
        constraints = t.data.get("constraints", {})
        assert episode_metadata(t)["max_tool_calls"] == str(constraints["max_tool_calls"]), t.id


def test_session_takes_max_turns_from_the_task(session, task):
    assert session.max_turns == task.data["constraints"]["max_turns"]


def test_toolbox_takes_max_tool_calls_from_the_task(session, task):
    assert session.sandbox.toolbox.max_tool_calls == task.data["constraints"]["max_tool_calls"]


# --------------------------------------------------------------------------- tool dispatch


def test_tool_returns_the_agent_visible_shape(session):
    out = session.tool("list_files", {"path": "repo"})
    assert set(out) == {"ok", "output", "done"}
    assert out["ok"] and not out["done"]


def test_tool_errors_are_reported_not_raised(session):
    out = session.tool("read_file", {"path": "nope.py"})
    assert out["ok"] is False
    assert "no such file" in out["output"]


def test_done_flips_on_submit(session):
    assert not session.done
    assert session.tool("submit")["done"]
    assert session.done


def test_done_flips_when_the_tool_budget_runs_out(session):
    session.sandbox.toolbox.max_tool_calls = 2
    session.tool("list_files")
    assert session.tool("list_files")["done"]


def test_forbidden_write_raises_the_gate_through_the_session(session):
    session.tool("write_file", {"path": "tests/test_x.py", "content": "pass"})
    assert GATE_TESTS_MODIFIED in session.sandbox.toolbox.gates


# --------------------------------------------------------------------------- grading


@pytest.mark.slow
def test_grade_is_idempotent(session):
    """5.6: eval and the RL reward must be one number, not two that happen to agree."""
    session.tool("submit")
    first = session.grade()
    assert session.grade() is first


@pytest.mark.slow
def test_grade_without_submit_raises_no_submit(session):
    assert GATE_NO_SUBMIT in session.grade().gates


@pytest.mark.slow
def test_tools_are_refused_once_the_episode_is_graded(session):
    session.tool("submit")
    session.grade()
    out = session.tool("write_file", {"path": "repo/humanize/number.py", "content": "x = 1"})
    assert out["ok"] is False
    assert "already graded" in out["output"]


# --------------------------------------------------------------------------- the store


def test_store_opens_one_session_per_rollout(tmp_path):
    store = SessionStore(parent=tmp_path)
    a = store.open("rollout-a", TASK_ID)
    b = store.open("rollout-b", TASK_ID)
    assert a is not b
    assert a.sandbox.root != b.sandbox.root
    store.close_all()


def test_reseeding_a_rollout_does_not_discard_its_work(tmp_path):
    """A retried /seed_session must not silently reset the sandbox mid-episode."""
    store = SessionStore(parent=tmp_path)
    first = store.open("rollout-a", TASK_ID)
    first.tool("write_file", {"path": "repo/humanize/number.py", "content": "# edited\n"})
    again = store.open("rollout-a", TASK_ID)
    assert again is first
    assert "# edited" in (first.sandbox.repo_dir / "humanize" / "number.py").read_text(encoding="utf-8")
    store.close_all()


def test_close_removes_the_session_and_its_sandbox(tmp_path):
    store = SessionStore(parent=tmp_path)
    session = store.open("rollout-a", TASK_ID)
    root = session.sandbox.root
    assert root.exists()
    store.close("rollout-a")
    assert store.get("rollout-a") is None
    assert not root.exists()


def test_get_on_an_unknown_rollout_returns_none(tmp_path):
    assert SessionStore(parent=tmp_path).get("never-seeded") is None
