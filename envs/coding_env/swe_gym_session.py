"""Episode session for Docker-backed SWE-Gym tasks."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from envs.coding_env.docker_sandbox import DockerSandbox
from envs.coding_env.swe_gym_verifier import (
    verify_candidate,
)
from envs.coding_env.verifier import (
    GATE_NO_SUBMIT,
    VerifyResult,
)
from rewards.compose import compose_reward


CandidateVerifier = Callable[..., dict[str, Any]]


@dataclass
class SWEGymEpisodeSession:
    """One SWE-Gym rollout backed by a DockerSandbox."""

    sandbox: DockerSandbox
    max_turns: int
    verifier: CandidateVerifier = verify_candidate
    _result: VerifyResult | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def task(self) -> Any:
        return self.sandbox.task

    @property
    def done(self) -> bool:
        box = self.sandbox.toolbox
        return (
            box.submitted
            or box.n_calls >= box.max_tool_calls
        )

    def tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute one existing Toolbox action."""

        if self._result is not None:
            return {
                "ok": False,
                "output": (
                    "Error: this episode is already graded."
                ),
                "done": True,
            }

        result = self.sandbox.toolbox.call(
            name,
            arguments,
        )

        return {
            "ok": result.ok,
            "output": result.output,
            "done": self.done,
        }

    def grade(self) -> VerifyResult:
        """Export and verify the candidate patch once."""

        if self._result is not None:
            return self._result

        box = self.sandbox.toolbox
        candidate_patch = self.sandbox.export_patch()

        with tempfile.TemporaryDirectory(
            prefix="swegym_candidate_"
        ) as temporary:
            candidate_path = (
                Path(temporary) / "candidate.patch"
            )

            candidate_path.write_text(
                candidate_patch,
                encoding="utf-8",
            )

            candidate = self.verifier(
                instance_id=self.task.instance_id,
                candidate_patch_path=candidate_path,
            )

        passed = int(
            candidate.get("tests_passed", 0)
        )
        collected = int(
            candidate.get("tests_total", 0)
        )
        floor_tests = int(
            candidate.get("floor_tests", 0)
        )

        gates = set(box.gates)

        if not box.submitted:
            gates.add(GATE_NO_SUBMIT)

        validation_status = candidate.get(
            "validation_status"
        )

        if validation_status != "candidate_valid":
            gates.add("candidate_verification_failed")

        ordered_gates = sorted(gates)

        if collected > 0:
            pass_fraction = passed / collected
        else:
            pass_fraction = 0.0

        if collected > floor_tests:
            reward = compose_reward(
                passed,
                collected,
                floor_tests,
                ordered_gates,
            )
        else:
            # Invalid verifier denominator must never
            # produce positive or neutral credit.
            reward = -1.0

            if (
                "candidate_verification_failed"
                not in ordered_gates
            ):
                ordered_gates.append(
                    "candidate_verification_failed"
                )
                ordered_gates.sort()

        per_test = _candidate_per_test(candidate)

        self._result = VerifyResult(
            task_id=self.task.instance_id,
            reward=reward,
            pass_fraction=pass_fraction,
            passed=passed,
            collected=collected,
            floor_tests=floor_tests,
            per_test=per_test,
            gates=ordered_gates,
            submitted=box.submitted,
            tool_calls=box.n_calls,
            limits_enforced={
                "docker_network_disabled": True,
                "docker_resource_limits": True,
            },
            stdout_tail=_candidate_stdout(candidate),
        )

        return self._result

    def close(self) -> None:
        self.sandbox.stop()


def _candidate_per_test(
    candidate: dict[str, Any],
) -> dict[str, str]:
    """Build a minimal per-test result mapping."""

    results: dict[str, str] = {}

    for key in [
        "fail_to_pass_result",
        "pass_to_pass_result",
    ]:
        group = candidate.get(key)

        if not isinstance(group, dict):
            continue

        tests = group.get("tests") or []
        exit_code = group.get("exit_code")

        status = (
            "passed"
            if exit_code == 0
            else "failed"
        )

        for test in tests:
            results[str(test)] = status

    return results


def _candidate_stdout(
    candidate: dict[str, Any],
) -> str:
    chunks: list[str] = []

    for key in [
        "fail_to_pass_result",
        "pass_to_pass_result",
    ]:
        group = candidate.get(key)

        if not isinstance(group, dict):
            continue

        for stream in ["stdout", "stderr"]:
            value = group.get(stream)

            if isinstance(value, str) and value:
                chunks.append(value)

    return "\n".join(chunks)[-4000:]
