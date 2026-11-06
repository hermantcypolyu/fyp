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

import hashlib
import json
import os
import platform
import stat
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from analysis import versions
from rewards.compose import REWARD_MIN

__all__ = [
    "MAX_PATCH_CHARS",
    "PRIVATE_MODE",
    "SCHEMA_PATH",
    "SCHEMA_V2_PATH",
    "SCHEMA_VERSION",
    "TERMINATION_REASONS",
    "TraceError",
    "build_row",
    "collect_patch",
    "read_jsonl",
    "read_private_jsonl",
    "secure_write_private_log",
    "validate_row",
    "write_jsonl",
    "write_private_jsonl",
]

SCHEMA_PATH = Path(__file__).resolve().parent / "trace_schema.json"
#: The v2 schema is kept in-tree, not just in git history: rows written before v3 still have
#: to be readable, and a reader that can only understand the current version silently turns
#: yesterday's results into unparseable files.
SCHEMA_V2_PATH = Path(__file__).resolve().parent / "trace_schema_v2.json"
SCHEMA_VERSION = "3"

#: Every rollout ends for exactly one of these reasons. `termination_reason` is the cause and
#: `result.gates` is the consequence; `max_turns` and `max_tool_calls` both fire `no_submit`,
#: and only the termination reason tells them apart.
TERMINATION_REASONS = ("submitted", "max_turns", "max_tool_calls", "timeout", "error")

#: Tool arguments and outputs are truncated here, not in the tools: the agent saw the full
#: text, and pretending otherwise would misrepresent the episode. `output_chars` keeps the
#: original length so context pressure stays measurable.
MAX_FIELD_CHARS = 600

#: The submitted diff gets a bigger budget than a tool argument: it is the single most
#: important artifact in the row, and clipping a patch at 600 chars would make it useless.
MAX_PATCH_CHARS = 20_000


class TraceError(Exception):
    """A row does not match trace_schema.json. The message names the field."""


@lru_cache(maxsize=2)
def _validator(schema_version: str) -> Draft202012Validator:
    """The validator for one schema version. Cached, because a run validates thousands of rows."""
    path = {"2": SCHEMA_V2_PATH, "3": SCHEMA_PATH}.get(schema_version)
    if path is None:
        raise TraceError(f"unknown schema_version {schema_version!r}; this code reads 2 and 3")
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_row(row: dict[str, Any]) -> None:
    """Validate against the schema named by the row's own `schema_version`.

    Dispatching on the row rather than on the current constant is what keeps old results
    readable: a v2 row is validated as a v2 row, and refusing it would make every historical
    run unanalysable the moment the schema moved on.
    """
    schema_version = row.get("schema_version")
    if schema_version is None:
        raise TraceError("trace row invalid at schema_version: field is required")
    errors = sorted(_validator(str(schema_version)).iter_errors(row), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(p) for p in first.absolute_path) or "<root>"
        raise TraceError(f"trace row invalid at {where}: {first.message}")


def _clip(text: str, limit: int = MAX_FIELD_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[+{len(text) - limit} chars]"


def _clip_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {k: _clip(v) if isinstance(v, str) else v for k, v in arguments.items()}


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


@dataclass
class Timer:
    """Wall-clock bracket for one trajectory."""

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


def _build_steps(
    tool_calls: list[Any], *, started_at: str, token_counts: list[int | None] | None
) -> list[dict[str, Any]]:
    """Turn the flat tool log into the ordered step list.

    One step per tool call. A turn that emitted no call is not reconstructible from the
    toolbox log, so it cannot be manufactured here; the harness that owns turns passes
    `token_counts` to annotate what it knows. What this function must never do is invent a
    step, because an invented step is a fabricated observation in the training data.
    """
    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
    steps: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        name = call.name
        if name == "submit":
            action_type = "submit"
        else:
            action_type = "tool_call"
        steps.append(
            {
                "step_id": index,
                "timestamp": _iso(started),
                "observation_summary": _clip(call.output),
                "action_type": action_type,
                "tool_name": name,
                "tool_arguments": _clip_arguments(call.arguments),
                "tool_result_summary": _clip(call.output),
                "ok": call.ok,
                "duration_ms": 0.0,
                "token_count": (token_counts[index] if token_counts and index < len(token_counts) else None),
            }
        )
    return steps


def collect_patch(task: Any, repo_dir: Path) -> str:
    """The unified diff of the agent's `repo/` against the pristine task tree.

    This exists because the sandbox is deleted on close and `write_file` arguments are
    clipped at `MAX_FIELD_CHARS`, so the change the agent made has no durable record
    anywhere else. The reference side is the task's own `repo/` in `benchmark/tasks/`, which
    is byte-identical to what was materialised (see `Sandbox.start`), not a second copy that
    could drift.

    Returns `""` when nothing changed, so the caller records `present: false` rather than a
    patch of zero hunks. A diff that fails to compute raises: a silently empty patch would be
    read as 'the agent changed nothing', which is a false statement about the rollout.
    """
    import difflib

    pristine_root = Path(task.path) / "repo"
    repo_dir = Path(repo_dir)
    if not pristine_root.is_dir():
        raise TraceError(f"{task.id}: no pristine repo/ at {pristine_root}, cannot diff the submission")

    def _read(path: Path) -> list[str]:
        return path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    chunks: list[str] = []
    pristine_files = {
        p.relative_to(pristine_root) for p in pristine_root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    working_files = {
        p.relative_to(repo_dir) for p in repo_dir.rglob("*") if p.is_file() and "__pycache__" not in p.parts
    }
    for rel in sorted(pristine_files | working_files):
        before_path = pristine_root / rel
        after_path = repo_dir / rel
        before = _read(before_path) if before_path.is_file() else []
        after = _read(after_path) if after_path.is_file() else []
        if before == after:
            continue
        chunks.extend(
            difflib.unified_diff(
                before,
                after,
                fromfile=f"a/repo/{rel.as_posix()}",
                tofile=f"b/repo/{rel.as_posix()}",
                lineterm="\n",
            )
        )
    return "".join(chunks)


def _build_candidate_patch(patch: str | None) -> dict[str, Any]:
    """Record the submitted diff. `sha256` is over the FULL text, so truncation cannot hide a match."""
    if not patch:
        return {"present": False, "sha256": None, "chars": 0, "truncated": False, "diff": None}
    return {
        "present": True,
        "sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        "chars": len(patch),
        "truncated": len(patch) > MAX_PATCH_CHARS,
        "diff": _clip(patch, MAX_PATCH_CHARS),
    }


def _termination_reason(
    *,
    result: Any,
    timed_out: bool,
    max_tool_calls: int,
    submitted: bool,
) -> str:
    """The cause of the end of the rollout, derived only from facts the run recorded.

    Deliberately not a read of `result.gates`: `no_submit` is the *consequence* of both
    `max_turns` and `max_tool_calls` exhausted, and using the gate alone would collapse two
    distinct failure modes into one. The budget that actually ran out is the cause.
    """
    if timed_out or (result is not None and getattr(result, "timed_out", False)):
        return "timeout"
    if submitted:
        return "submitted"
    tool_calls_used = int(getattr(result, "tool_calls", 0)) if result is not None else 0
    if max_tool_calls > 0 and tool_calls_used >= max_tool_calls:
        return "max_tool_calls"
    return "max_turns"


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
    checkpoint_id: str | None = None,
    random_seed: int | None = None,
    prompt_config_version: str | None = None,
    candidate_patch: str | None = None,
    token_usage: dict[str, int] | None = None,
    step_token_counts: list[int | None] | None = None,
    turns_used: int | None = None,
    reward_components: dict[str, float | None] | None = None,
    error: BaseException | None = None,
    validate: bool = True,
) -> dict[str, Any]:
    """Assemble one v3 trace row from a finished rollout.

    `task` is a `benchmark.tools.task_loader.Task`, `result` a
    `envs.coding_env.verifier.VerifyResult`, `tool_calls` the toolbox log.

    Every optional parameter defaults to an honest null rather than a plausible value. A
    Seed or a checkpoint that a caller did not supply is recorded as unknown, because a
    fabricated reproducibility field is worse than an absent one: it makes an unreproducible
    run look reproducible.
    """
    started, finished, duration = timer.stop()
    constraints = task.data.get("constraints", {}) or {}
    provenance = task.data.get("provenance", {}) or {}

    calls = list(tool_calls)
    steps = _build_steps(calls, started_at=started, token_counts=step_token_counts)
    max_tool_calls = int(constraints.get("max_tool_calls", 40))

    if error is not None or result is None:
        termination = "error"
    else:
        termination = _termination_reason(
            result=result,
            timed_out=False,
            max_tool_calls=max_tool_calls,
            submitted=bool(result.submitted),
        )

    if result is None:
        # A rollout that raised before grading still gets a row. Without this branch the
        # crashed rollouts vanish from the dataset and a GRPO group of them averages to a
        # zero-variance 0.0 that is indistinguishable from "ran fine, achieved nothing".
        result_block: dict[str, Any] = {
            "reward": REWARD_MIN,
            "pass_fraction": 0.0,
            "tests_passed": 0,
            "tests_total": 0,
            "floor_tests": 0,
            "gates": [],
            "submitted": False,
        }
        hidden = {"declared": task.n_tests, "passed": 0, "floor": 0, "failed": task.n_tests}
        tool_calls_used = len(calls)
    else:
        result_block = {
            "reward": result.reward,
            "pass_fraction": result.pass_fraction,
            "tests_passed": result.passed,
            "tests_total": result.collected,
            "floor_tests": result.floor_tests,
            "gates": list(result.gates),
            "submitted": result.submitted,
        }
        hidden = {
            "declared": task.n_tests,
            "passed": result.passed,
            "floor": result.floor_tests,
            # A count only. The nodeids stay in rollouts_private.jsonl: a dashboard that
            # dumps this row must not be able to name the tests the model failed.
            "failed": max(result.collected - result.passed, 0),
        }
        tool_calls_used = len(calls)

    prompt_tokens = None if token_usage is None else token_usage.get("prompt")
    completion_tokens = None if token_usage is None else token_usage.get("completion")
    total_tokens = None if token_usage is None else token_usage.get("total")

    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "rollout_id": rollout_id,
        "termination_reason": termination,
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
        "agent": {
            "kind": agent_kind,
            "model": model,
            "checkpoint_id": checkpoint_id,
            "random_seed": random_seed,
            "prompt_config_version": prompt_config_version,
            "transport": transport,
        },
        "budgets": {
            "max_turns": int(constraints.get("max_turns", 20)),
            "max_tool_calls": int(constraints.get("max_tool_calls", 40)),
            "tool_calls_used": tool_calls_used,
        },
        "turns": {
            "max_turns": int(constraints.get("max_turns", 20)),
            # A turn without a tool call is invisible to the toolbox log, so the caller's
            # count wins when it has one; len(calls) is a documented lower bound.
            "turns_used": turns_used if turns_used is not None else len(calls),
        },
        "token_usage": {"prompt": prompt_tokens, "completion": completion_tokens, "total": total_tokens},
        "steps": steps,
        "candidate_patch": _build_candidate_patch(candidate_patch),
        "result": result_block,
        "reward_components": _build_reward_components(result_block["reward"], reward_components),
        "hidden_tests": hidden,
        "error": None if error is None else {"type": type(error).__name__, "message": _clip(str(error))},
        "limits_enforced": dict(getattr(result, "limits_enforced", {"wall_clock_s": False, "network": False})),
        "versions": {
            "repo_commit": versions.repo_commit(),
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.release()} ({sys.platform})",
            **versions.resolve_all(),
        },
    }
    if validate:
        _check_row_invariants(row)
        validate_row(row)
    return row


def _check_row_invariants(row: dict[str, Any]) -> None:
    """Cross-field rules a JSON Schema cannot express.

    Every one of these encodes a way the row could claim two contradictory things at once,
    which is worse than a missing field: a missing field is visibly missing, whereas a
    contradiction survives into a table and has to be discovered.
    """
    reason = row["termination_reason"]
    error = row["error"]

    if reason == "error" and error is None:
        raise TraceError("termination_reason is 'error' but `error` is null: nothing names the failure")
    if reason != "error" and error is not None:
        raise TraceError(f"`error` is set but termination_reason is {reason!r}")

    if reason == "submitted" and not row["result"]["submitted"]:
        raise TraceError("termination_reason is 'submitted' but result.submitted is false")

    if row["reward_components"]["final"] != row["result"]["reward"]:
        raise TraceError("reward_components.final disagrees with result.reward")

    if row["budgets"]["tool_calls_used"] != len(row["steps"]):
        raise TraceError("budgets.tool_calls_used disagrees with len(steps): a step is a tool call")

    if row["turns"]["turns_used"] < len(row["steps"]):
        raise TraceError("turns_used cannot be smaller than the number of steps taken")

    patch = row["candidate_patch"]
    if patch["present"] and not patch["diff"]:
        raise TraceError("candidate_patch.present is true but the diff is empty")
    if not patch["present"] and patch["sha256"] is not None:
        raise TraceError("candidate_patch.present is false but a sha256 was recorded")


def _build_reward_components(final: float, components: dict[str, float | None] | None) -> dict[str, Any]:
    """reward_v2 slots. Unshipped components stay None - not 0.0, which would read as 'scored zero'."""
    supplied = components or {}
    result = {
        "correctness": supplied.get("correctness"),
        "progress": supplied.get("progress"),
        "validity": supplied.get("validity"),
        "tool": supplied.get("tool"),
        # `final` must equal result.reward, so the fused total can never drift from the
        # decomposition that is supposed to explain it.
        "final": final,
    }
    if result["final"] != final:
        raise TraceError("reward_components.final must equal result.reward")
    return result


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


def write_private_jsonl(
    rows: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    path: Path,
    *,
    append: bool = False,
) -> Path:
    """Write the private verification detail, keyed to the public rows.

    `rows` is `(public_row, private_payload)` pairs. The payload carries `per_test` nodeid ->
    outcome, which `rollouts.jsonl` deliberately does not. Two files rather than one because
    the analysis needs the nodeids and everything that renders to a human or to a live agent
    must not be able to reach them; the join key is `(run_id, rollout_id)`.

    There is no JSON Schema here on purpose. The public row is the contract; the private file
    is a sidecar whose only stability requirement is that it stays joinable, and pinning its
    shape with a second schema would create a second thing to migrate for no benefit.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8", newline="\n") as fh:
        for public_row, payload in rows:
            record = {
                "run_id": public_row["run_id"],
                "rollout_id": public_row["rollout_id"],
                "task_id": public_row["task"]["id"],
                "per_test": dict(payload.get("per_test") or {}),
                "gold_patch_sha256": payload.get("gold_patch_sha256"),
            }
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def read_private_jsonl(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load the private sidecar as `(run_id, rollout_id) -> record`."""
    return {(row["run_id"], row["rollout_id"]): row for row in read_jsonl(path)}


#: Owner read/write only. The private log holds raw tool arguments and observations, which
#: can contain the contents of the task's repository - including a gold patch the agent
#: happened to read. Nothing but its owner may open it.
PRIVATE_MODE = 0o600


def secure_write_private_log(path: Path, records: Iterable[dict[str, Any]], *, append: bool = False) -> Path:
    """Write the private rollout log with owner-only permissions, verified after opening.

    `write_private_jsonl` above is for the verifier's per-test sidecar. This is for the
    collector's private log, whose record shape is the collector's own and which must not be
    reshaped: Phase 6E artifacts are already on disk in that format, and rewriting them to a
    new shape would destroy the evidence they exist to preserve.

    **Why not just `open(..., "w")`.** That is what Phase 6E did, and the file landed at
    0664 because the mode came from the process umask. A privacy requirement that is
    satisfied only when someone happens to have a restrictive umask is not a requirement.
    Two things are therefore done explicitly:

    1. `os.open` with `mode=0o600` so the file is *created* unreadable to group/other - there
       is no window in which the content exists with wider permissions.
    2. `os.fchmod` on the open descriptor, which also narrows an existing file whose mode
       drifted. This is what makes a resume safe: appending must not inherit a broad mode
       from a file written before this fix.

    The mode is read back off the descriptor and asserted, because the umask can still
    remove bits, and a silent fallback to 0644 is exactly the bug being fixed.
    """

    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    fd = os.open(path, flags, PRIVATE_MODE)
    try:
        # Narrow explicitly: O_CREAT's mode is masked by the umask and does not apply to a
        # file that already existed.
        os.fchmod(fd, PRIVATE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n", closefd=False) as fh:
            for record in records:
                fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        actual = stat.S_IMODE(os.fstat(fd).st_mode)
    finally:
        os.close(fd)

    if actual != PRIVATE_MODE:
        raise TraceError(
            f"private log {path} has mode {actual:04o}, expected {PRIVATE_MODE:04o}; "
            "refusing to leave a readable private log on disk"
        )
    return path
