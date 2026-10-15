"""Episode session state: everything the resources server does, minus the HTTP.

`nemo_gym` does not import on Windows out of the box - `nemo_gym/server_utils.py` has a
top-level `import resource`, which is POSIX-only. B0's `_vendor/win_compat/` shim works
around it, and `app.py` has been driven end-to-end through Starlette's TestClient with that
shim in place. But the shim lives outside this repo and is not part of `uv sync`, so the
default developer environment here cannot import `app.py` at all.

Hence the split: everything below is framework-free and covered by
`envs/tests/test_session.py`, and `app.py` is a routing table over it - request model in,
`session.tool(...)` out. The part that depends on an out-of-tree shim is a few dozen FastAPI
decorators, not the environment's behaviour.

See docs/decisions/0002-nemo-gym-is-an-optional-extra.md and docs/nemo_gym_notes.md 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from shutil import rmtree
from typing import Any

from benchmark.tools.task_loader import SPLITS, TASKS_ROOT, Task, TaskError, load_task
from envs.coding_env.limits import ResourceLimits
from envs.coding_env.sandbox import Sandbox
from envs.coding_env.verifier import VerifyResult, verify_sandbox

__all__ = ["EpisodeSession", "SessionStore", "episode_metadata", "find_task"]


@cache
def find_task(task_id: str, split: str | None = None, root: Path = TASKS_ROOT) -> Task:
    """Resolve a task id to a `Task`. Cached: a rollout batch hits the same ids repeatedly."""
    for candidate in (split,) if split else SPLITS:
        path = root / candidate / task_id
        if (path / "task.yaml").is_file():
            return load_task(path)
    raise TaskError(f"no task {task_id!r} under {root}" + (f" in split {split!r}" if split else ""))


def episode_metadata(task: Task) -> dict[str, str]:
    """The `constraints:` block, in the shape that rides on `responses_create_params`.

    The two budgets are enforced in different places, for a reason.

    `max_tool_calls` is enforced **server-side** by the toolbox, because it is a property of
    the environment: an agent must not be able to opt out of it, and the resources server is
    the only component that sees every call.

    `max_turns` is enforced **agent-side**, because a turn is a loop iteration and only the
    loop can count them. The built-in `simple_agent` reads its cap from process-wide config,
    which cannot vary per task - and the corpus does vary (20/25/30). Rather than fork the
    loop to fix that, `CodingAgent` reads this metadata and hands the built-in loop the right
    number. Values are strings because the Responses API's `metadata` is a string map.
    """
    constraints = task.data.get("constraints", {}) or {}
    repo = task.data.get("repo", {}) or {}
    entrypoint_hint = repo.get("entrypoint_hint")

    metadata = {
        "task_id": task.id,
        "split": task.split,
        "max_turns": str(constraints.get("max_turns", 20)),
        "max_tool_calls": str(constraints.get("max_tool_calls", 40)),
    }

    if entrypoint_hint:
        metadata["entrypoint_hint"] = f"repo/{entrypoint_hint}"

    return metadata


@dataclass
class EpisodeSession:
    """One rollout: a sandbox, the toolbox bound to it, and at most one graded result."""

    sandbox: Sandbox
    max_turns: int
    limits: ResourceLimits | None = None
    _result: VerifyResult | None = field(default=None, init=False, repr=False)

    @classmethod
    def open(cls, task: Task, parent: Path | None = None, limits: ResourceLimits | None = None) -> EpisodeSession:
        constraints = task.data.get("constraints", {}) or {}
        return cls(
            sandbox=Sandbox.start(task, parent),
            max_turns=int(constraints.get("max_turns", 20)),
            limits=limits,
        )

    @property
    def task(self) -> Task:
        return self.sandbox.task

    @property
    def done(self) -> bool:
        """True once the agent has submitted or spent its tool-call budget."""
        box = self.sandbox.toolbox
        return box.submitted or box.n_calls >= box.max_tool_calls

    def tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run one tool call. The dict is what the agent sees as the tool message."""
        if self._result is not None:
            return {"ok": False, "output": "Error: this episode is already graded.", "done": True}
        result = self.sandbox.toolbox.call(name, arguments)
        return {"ok": result.ok, "output": result.output, "done": self.done}

    def grade(self) -> VerifyResult:
        """Score the episode. Idempotent - the second call returns the first result.

        Idempotence is not an optimisation. 5.6 requires evaluation and the RL reward to be
        the same number; caching makes that true by construction rather than by two calls
        happening to agree.
        """
        if self._result is None:
            self._result = verify_sandbox(self.sandbox, limits=self.limits)
        return self._result

    def close(self) -> None:
        self.sandbox.stop()


@dataclass
class SessionStore:
    """Rollout id -> live episode. One instance lives on the resources server."""

    parent: Path | None = None
    limits: ResourceLimits | None = None
    sessions: dict[str, EpisodeSession] = field(default_factory=dict)

    def open(self, session_id: str, task_id: str, split: str | None = None) -> EpisodeSession:
        """Idempotent: re-seeding an existing rollout must not silently discard its work."""
        if session_id in self.sessions:
            return self.sessions[session_id]
        task = find_task(task_id, split)
        parent = None if self.parent is None else Path(self.parent) / session_id
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        session = EpisodeSession.open(task, parent=parent, limits=self.limits)
        self.sessions[session_id] = session
        return session

    def get(self, session_id: str) -> EpisodeSession | None:
        return self.sessions.get(session_id)

    def close(self, session_id: str) -> None:
        session = self.sessions.pop(session_id, None)
        if session is None:
            return
        session.close()
        if self.parent is not None:  # the per-rollout directory this store created
            rmtree(Path(self.parent) / session_id, ignore_errors=True)

    def close_all(self) -> None:
        for session_id in list(self.sessions):
            self.close(session_id)
