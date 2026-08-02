"""Resources server: the six tools over HTTP, plus `POST /verify`.

This is the only module in the project that imports `nemo_gym`, and it is deliberately
thin - a routing table over `envs.coding_env.session`, which holds all the behaviour and is
tested without the framework. The reason is not taste: `nemo_gym/server_utils.py` has a
top-level `import resource`, so this file needs either POSIX or B0's out-of-tree
`_vendor/win_compat/` shim, neither of which `uv sync` provides on Windows (ADR 0001,
docs/nemo_gym_notes.md 7). Keeping logic out of here is what keeps the environment testable
in the default developer environment.

It has been driven end-to-end with the shim: `scripts/smoke_one_task.py --transport gym`
does `/seed_session` -> six tool routes -> `/verify` against this app through Starlette's
TestClient, and scores the gold solution 1.0.

⭐ 5.6's critical invariant lives here: `/verify` calls `EpisodeSession.grade()`, which calls
`envs.coding_env.verifier.verify()` - the same function `scripts/verify_gold_patches.py`
and every offline evaluation call. There is no second scoring path.

# Pattern adapted from: nemo-gym/resources_servers/example_session_state_mgmt/app.py
# (per-rollout state keyed by SESSION_ID_KEY, seed_session/verify overrides)
# and nemo-gym/resources_servers/example_multi_step/app.py (tool routes on
# SimpleResourcesServer.setup_webserver) @ commit ec5a8136046aa0dd0c1525313654502f2fb7c42e
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from pydantic import BaseModel, ConfigDict, Field

from envs.coding_env.session import SessionStore

__all__ = ["CodingResourcesServer"]


class CodingResourcesServerConfig(BaseResourcesServerConfig):
    #: Where sandboxes are built. None -> a temp directory per rollout.
    sandbox_root: str | None = None
    #: Grading is stateful - the reward depends on a sandbox that only exists during the
    #: rollout - so a verify replayed from a trajectory alone cannot reproduce it.
    REVERIFY_MODE = ReverifyMode.UNSUPPORTED


# --------------------------------------------------------------------------- tool payloads


class ListFilesRequest(BaseModel):
    path: str = "."


class ReadFileRequest(BaseModel):
    path: str
    start: int | None = None
    end: int | None = None


class WriteFileRequest(BaseModel):
    path: str
    content: str


class EditFileRequest(BaseModel):
    path: str
    old_str: str
    new_str: str


class NoArgsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # a small model often emits `{}` or stray keys


class ToolResponse(BaseModel):
    ok: bool
    output: str
    done: bool


# --------------------------------------------------------------------------- lifecycle


class CodingSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")

    task_id: str
    split: str | None = None


class CodingVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    task_id: str
    split: str | None = None


class CodingVerifyResponse(BaseVerifyResponse):
    """`reward` is inherited. The rest is the structured result 5.6 asks for."""

    model_config = ConfigDict(extra="allow")

    pass_fraction: float
    tests_passed: int
    tests_total: int
    per_test: dict[str, str]
    gates: list[str]
    submitted: bool
    tool_calls: int
    limits_enforced: dict[str, bool]


class CodingResourcesServer(SimpleResourcesServer):
    config: CodingResourcesServerConfig
    store: SessionStore = Field(default_factory=SessionStore)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/list_files")(self.list_files)
        app.post("/read_file")(self.read_file)
        app.post("/write_file")(self.write_file)
        app.post("/edit_file")(self.edit_file)
        app.post("/run_tests")(self.run_tests)
        app.post("/submit")(self.submit)
        return app

    # ------------------------------------------------------------------ lifecycle

    async def seed_session(self, request: Request, body: CodingSeedSessionRequest) -> BaseSeedSessionResponse:
        if self.config.sandbox_root:
            self.store.parent = self.config.sandbox_root
        self.store.open(request.session[SESSION_ID_KEY], body.task_id, body.split)
        return BaseSeedSessionResponse()

    async def verify(self, request: Request, body: CodingVerifyRequest) -> CodingVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        session = self.store.get(session_id)
        if session is None:
            # Loud, not a silent 0.0. A missing sandbox means the rollout was never seeded or
            # was already reaped; scoring it as "the agent failed" would put a fabricated row
            # in the dataset and quietly depress every aggregate computed from it.
            raise HTTPException(status_code=409, detail=f"no live session for rollout {session_id}")
        try:
            result = session.grade()
        finally:
            self.store.close(session_id)
        return CodingVerifyResponse(
            **body.model_dump(),
            reward=result.reward,
            pass_fraction=result.pass_fraction,
            tests_passed=result.passed,
            tests_total=result.collected,
            per_test=result.per_test,
            gates=result.gates,
            submitted=result.submitted,
            tool_calls=result.tool_calls,
            limits_enforced=result.limits_enforced,
        )

    # ------------------------------------------------------------------ the six tools

    def _tool(self, request: Request, name: str, arguments: dict[str, Any]) -> ToolResponse:
        session = self.store.get(request.session[SESSION_ID_KEY])
        if session is None:
            raise HTTPException(status_code=409, detail="no live session; POST /seed_session first")
        return ToolResponse(**session.tool(name, arguments))

    async def list_files(self, request: Request, body: ListFilesRequest) -> ToolResponse:
        return self._tool(request, "list_files", body.model_dump())

    async def read_file(self, request: Request, body: ReadFileRequest) -> ToolResponse:
        return self._tool(request, "read_file", body.model_dump())

    async def write_file(self, request: Request, body: WriteFileRequest) -> ToolResponse:
        return self._tool(request, "write_file", body.model_dump())

    async def edit_file(self, request: Request, body: EditFileRequest) -> ToolResponse:
        return self._tool(request, "edit_file", body.model_dump())

    async def run_tests(self, request: Request, body: NoArgsRequest) -> ToolResponse:
        return self._tool(request, "run_tests", {})

    async def submit(self, request: Request, body: NoArgsRequest) -> ToolResponse:
        return self._tool(request, "submit", {})


if __name__ == "__main__":
    CodingResourcesServer.run_webserver()
