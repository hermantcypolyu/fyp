"""Agent harness: `simple_agent` with a per-task turn budget. Nothing else.

The built-in `SimpleAgent` already implements the whole tool-augmented loop - model call,
tool dispatch, malformed-argument recovery, usage accounting, cookie propagation. Rewriting
that would be a way to reintroduce its bugs, so this subclass adds exactly one behaviour and
inherits the rest.

**What it adds.** The built-in reads its step cap from `config.max_steps`, which is
process-wide. Our corpus varies it per task (20 / 25 / 30 turns, per `task.yaml`
`constraints.max_turns`), and a shared server serves every task, so a single config value
cannot be right. `responses()` therefore reads the per-rollout cap out of the request's
`metadata` and hands the built-in loop a config copy carrying that number. The loop itself is
untouched.

A copy rather than a mutation: the server is one object serving concurrent rollouts, and
assigning to `self.config.max_steps` would let one task's budget leak into another's.

`max_tool_calls` is *not* handled here - it is enforced by the resources server, because an
agent must not be able to opt out of an environment limit.

# Pattern adapted from: nemo-gym/responses_api_agents/simple_agent/app.py
# (SimpleAgent.responses / .run) @ commit ec5a8136046aa0dd0c1525313654502f2fb7c42e
"""

from __future__ import annotations

from fastapi import Request, Response
from nemo_gym.base_responses_api_agent import Body
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig

__all__ = ["CodingAgent", "CodingAgentConfig"]


class CodingAgentConfig(SimpleAgentConfig):
    #: Backstop used when a row carries no `max_turns`. Deliberately above the corpus
    #: maximum (30) so that a missing value degrades to "generous", not to "truncated" -
    #: a silently short episode looks exactly like a model that gave up.
    max_steps: int = 40

    #: Scaffold a complete coding episode for RL infrastructure testing.
    first_tool_name: str | None = "list_files"
    second_tool_name: str | None = "read_file"
    third_tool_name: str | None = "edit_file"
    fourth_tool_name: str | None = "run_tests"
    fifth_tool_name: str | None = "submit"


class CodingAgent(SimpleAgent):
    config: CodingAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),  # noqa: B008 - NeMo Gym's route signature
    ) -> NeMoGymResponse:
        cap = _max_turns(body) or self.config.max_steps
        scoped = self.model_copy(update={"config": self.config.model_copy(update={"max_steps": cap})})

        # metadata is environment-side information used to select the per-task
        # turn budget. Re-validate from a dump that excludes metadata so the
        # field is genuinely unset. ServerClient serialises with
        # exclude_unset=True; setting metadata=None would still forward
        # "metadata": null to the pinned vLLM endpoint, which rejects it.
        model_body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
            body.model_dump(exclude={"metadata", "service_tier", "store"})
        )

        return await SimpleAgent.responses(scoped, request, response, model_body)


def _max_turns(body: NeMoGymResponseCreateParamsNonStreaming) -> int | None:
    """`constraints.max_turns` from task.yaml, put on the row by the dataset builder.

    The Responses API's `metadata` is a string map, so the value arrives as a string. A
    malformed one is ignored rather than raised on: losing the cap costs a few wasted turns,
    while raising would fail the whole rollout over a data-entry mistake.
    """
    raw = (getattr(body, "metadata", None) or {}).get("max_turns")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


if __name__ == "__main__":
    CodingAgent.run_webserver()
