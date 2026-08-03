"""Rollout traces: one JSON line per episode, validated against `trace_schema.json`.

The trace is the only durable record of a rollout. The sandbox is deleted, the model is
stateless, and the reward is one float - if a number in the dissertation cannot be traced
back to a row here, it cannot be defended.

Two decisions worth stating.

**The row is self-contained.** Task category, provenance, budgets, the commits of every
moving part and which resource caps the host actually applied are all copied in, rather than
looked up later from the corpus. A trace that needs `benchmark/tasks/` at its original
revision to be interpreted stops being interpretable the moment a task is edited.

**Validation happens on write, not on read.** A malformed row discovered during analysis is
a run that has to be repeated; discovered at write time it is a bug fixed in seconds. The
cost is one schema check per episode, against a rollout that just spent seconds running
pytest.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from analysis import versions

__all__ = ["SCHEMA_PATH", "SCHEMA_VERSION", "TraceError", "build_row", "read_jsonl", "validate_row", "write_jsonl"]

SCHEMA_PATH = Path(__file__).resolve().parent / "trace_schema.json"
SCHEMA_VERSION = "2"

#: Tool arguments and outputs are truncated here, not in the tools: the agent saw the full
#: text, and pretending otherwise would misrepresent the episode. `output_chars` keeps the
#: original length so context pressure stays measurable.
MAX_FIELD_CHARS = 600


class TraceError(Exception):
    """A row does not match trace_schema.json. The message names the field."""


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_row(row: dict[str, Any]) -> None:
    errors = sorted(_validator().iter_errors(row), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(p) for p in first.absolute_path) or "<root>"
        raise TraceError(f"trace row invalid at {where}: {first.message}")


def _clip(text: str) -> str:
    if len(text) <= MAX_FIELD_CHARS:
        return text
    return f"{text[:MAX_FIELD_CHARS]}...[+{len(text) - MAX_FIELD_CHARS} chars]"


def _clip_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {k: _clip(v) if isinstance(v, str) else v for k, v in arguments.items()}


@dataclass
class Timer:
    """Wall-clock bracket for one episode."""

    started_at: datetime

    @classmethod
    def start(cls) -> Timer:
        return cls(started_at=datetime.now(UTC))

    def stop(self) -> tuple[str, str, float]:
        finished = datetime.now(UTC)
        return (
            self.started_at.isoformat().replace("+00:00", "Z"),
            finished.isoformat().replace("+00:00", "Z"),
            round((finished - self.started_at).total_seconds(), 3),
        )


def build_row(
    *,
    run_id: str,
    rollout_id: str,
    task: Any,
    result: Any,
    tool_calls: Iterable[Any],
    timer: Timer,
    transport: str = "direct",
    agent_kind: str = "policy",
    model: str | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Assemble one trace row from a finished episode.

    `task` is a `benchmark.tools.task_loader.Task`, `result` a
    `envs.coding_env.verifier.VerifyResult`, `tool_calls` the toolbox log.
    """
    started, finished, duration = timer.stop()
    constraints = task.data.get("constraints", {}) or {}
    provenance = task.data.get("provenance", {}) or {}

    calls = list(tool_calls)
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "rollout_id": rollout_id,
        "started_at": started,
        "finished_at": finished,
        "duration_s": duration,
        "task": {
            "id": task.id,
            "split": task.split,
            "category": task.category,
            "difficulty": task.difficulty,
            "provenance_source": str(provenance.get("source", "unknown")),
            "n_tests_declared": task.n_tests,
        },
        "agent": {"kind": agent_kind, "model": model, "transport": transport},
        "budgets": {
            "max_turns": int(constraints.get("max_turns", 20)),
            "max_tool_calls": int(constraints.get("max_tool_calls", 40)),
            "tool_calls_used": len(calls),
        },
        "tool_calls": [
            {
                "index": i,
                "name": call.name,
                "arguments": _clip_arguments(call.arguments),
                "ok": call.ok,
                "output": _clip(call.output),
                "output_chars": len(call.output),
            }
            for i, call in enumerate(calls)
        ],
        "result": {
            "reward": result.reward,
            "pass_fraction": result.pass_fraction,
            "tests_passed": result.passed,
            "tests_total": result.collected,
            "floor_tests": result.floor_tests,
            "gates": list(result.gates),
            "submitted": result.submitted,
            "per_test": dict(result.per_test),
        },
        "limits_enforced": dict(result.limits_enforced),
        "versions": {
            "repo_commit": versions.repo_commit(),
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()} ({sys.platform})",
            **versions.resolve_all(),
        },
    }
    if validate:
        validate_row(row)
    return row


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path, *, append: bool = False) -> Path:
    """Append-safe JSONL. Rows are re-validated: a hand-built row must not slip through."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            validate_row(row)
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise TraceError(f"{path}:{number}: not valid JSON - {e}") from e
