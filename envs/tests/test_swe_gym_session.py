"""SWE-Gym episode-session integration tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from envs.coding_env.verifier import GATE_NO_SUBMIT, VerifyResult


INSTANCE_ID = "getmoto__moto-5699"


class FakeToolbox:
    def __init__(self) -> None:
        self.submitted = False
        self.max_tool_calls = 40
        self.n_calls = 0
        self.gates: set[str] = set()

    def call(self, name, arguments=None):
        self.n_calls += 1

        if name == "submit":
            self.submitted = True

        return SimpleNamespace(
            ok=True,
            output=f"called {name}",
        )


class FakeDockerSandbox:
    def __init__(self, root: Path) -> None:
        self.task = SimpleNamespace(
            instance_id=INSTANCE_ID,
        )
        self.root = root
        self.repo_dir = root / "repo"
        self.repo_dir.mkdir(parents=True)
        self.toolbox = FakeToolbox()
        self.stopped = False

    def export_patch(self) -> str:
        return (
            "diff --git a/example.py b/example.py\n"
            "--- a/example.py\n"
            "+++ b/example.py\n"
            "@@ -1 +1 @@\n"
            "-value = 1\n"
            "+value = 2\n"
        )

    def stop(self) -> None:
        self.stopped = True


def candidate_result() -> dict:
    return {
        "candidate_patch_applied": True,
        "tests_passed": 2,
        "tests_total": 3,
        "floor_tests": 2,
        "tests_above_floor": 0,
        "candidate_solved": False,
        "validation_status": "candidate_valid",
        "container_cleaned": True,
        "fail_to_pass_result": {
            "tests": ["hidden_f2p"],
            "exit_code": 1,
        },
        "pass_to_pass_result": {
            "tests": ["hidden_p2p_1", "hidden_p2p_2"],
            "exit_code": 0,
        },
    }


def test_tool_returns_existing_agent_visible_shape(
    tmp_path,
):
    from envs.coding_env.swe_gym_session import (
        SWEGymEpisodeSession,
    )

    sandbox = FakeDockerSandbox(tmp_path / "work")
    session = SWEGymEpisodeSession(
        sandbox=sandbox,
        max_turns=20,
        verifier=lambda *_args, **_kwargs: candidate_result(),
    )

    output = session.tool(
        "list_files",
        {"path": "repo"},
    )

    assert set(output) == {
        "ok",
        "output",
        "done",
    }
    assert output["ok"]
    assert not output["done"]


def test_submit_marks_episode_done(tmp_path):
    from envs.coding_env.swe_gym_session import (
        SWEGymEpisodeSession,
    )

    sandbox = FakeDockerSandbox(tmp_path / "work")
    session = SWEGymEpisodeSession(
        sandbox=sandbox,
        max_turns=20,
        verifier=lambda *_args, **_kwargs: candidate_result(),
    )

    assert not session.done
    assert session.tool("submit")["done"]
    assert session.done


def test_grade_converts_candidate_result_to_verify_result(
    tmp_path,
):
    from envs.coding_env.swe_gym_session import (
        SWEGymEpisodeSession,
    )

    sandbox = FakeDockerSandbox(tmp_path / "work")
    sandbox.toolbox.submitted = True

    session = SWEGymEpisodeSession(
        sandbox=sandbox,
        max_turns=20,
        verifier=lambda *_args, **_kwargs: candidate_result(),
    )

    result = session.grade()

    assert isinstance(result, VerifyResult)
    assert result.task_id == INSTANCE_ID
    assert result.passed == 2
    assert result.collected == 3
    assert result.floor_tests == 2
    assert result.pass_fraction == pytest.approx(
        2 / 3
    )
    assert result.reward == 0.0
    assert result.submitted
    assert result.gates == []


def test_empty_patch_is_still_verifiable(
    tmp_path,
):
    from envs.coding_env.swe_gym_session import (
        SWEGymEpisodeSession,
    )

    calls = []

    def verifier(
        *,
        instance_id,
        candidate_patch_path,
    ):
        calls.append(
            {
                "instance_id": instance_id,
                "patch_bytes": (
                    candidate_patch_path.stat().st_size
                ),
            }
        )

        return candidate_result()

    sandbox = FakeDockerSandbox(
        tmp_path / "work"
    )

    sandbox.export_patch = lambda: ""
    sandbox.toolbox.submitted = True

    session = SWEGymEpisodeSession(
        sandbox=sandbox,
        max_turns=20,
        verifier=verifier,
    )

    result = session.grade()

    assert calls == [
        {
            "instance_id": INSTANCE_ID,
            "patch_bytes": 0,
        }
    ]

    assert result.passed == 2
    assert result.collected == 3
    assert result.floor_tests == 2
    assert result.reward == 0.0
    assert result.submitted is True
    assert result.gates == []


def test_grade_without_submit_applies_no_submit_gate(
    tmp_path,
):
    from envs.coding_env.swe_gym_session import (
        SWEGymEpisodeSession,
    )

    sandbox = FakeDockerSandbox(tmp_path / "work")

    session = SWEGymEpisodeSession(
        sandbox=sandbox,
        max_turns=20,
        verifier=lambda *_args, **_kwargs: candidate_result(),
    )

    result = session.grade()

    assert GATE_NO_SUBMIT in result.gates
    assert result.reward == -1.0
    assert not result.submitted


def test_grade_is_idempotent_and_close_stops_sandbox(
    tmp_path,
):
    from envs.coding_env.swe_gym_session import (
        SWEGymEpisodeSession,
    )

    calls = 0

    def verifier(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return candidate_result()

    sandbox = FakeDockerSandbox(tmp_path / "work")
    sandbox.toolbox.submitted = True

    session = SWEGymEpisodeSession(
        sandbox=sandbox,
        max_turns=20,
        verifier=verifier,
    )

    first = session.grade()
    second = session.grade()

    assert second is first
    assert calls == 1

    session.close()
    assert sandbox.stopped
