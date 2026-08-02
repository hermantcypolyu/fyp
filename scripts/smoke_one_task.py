"""Drive ONE task end-to-end through the environment and print the reward.

The "agent" here is scripted, not a model: it lists the repo, reads the entrypoint, writes
the gold solution **through the real `write_file` tool**, runs the smoke tests and submits.
That is the point - a model would test the model, whereas this tests the environment. If
the reward is not 1.0, something in the tool surface, the sandbox, the write globs or the
verifier is broken, and no amount of prompt engineering later will fix it.

Two transports, the same episode:

    --transport direct   in-process, no framework   -> works on any OS
    --transport gym      over the FastAPI app       -> needs the nemo-gym stack

`direct` is the one wired into `make smoke`, because the environment must stay checkable
without NeMo Gym installed (ADR 0002). `gym` is `make smoke-gym`, and exists to prove the
HTTP layer in `envs/coding_env/app.py` is wired correctly - the routes, the session
middleware and `POST /verify` - which the direct path never touches.

    uv run python scripts/smoke_one_task.py --limit 1
    uv run python scripts/smoke_one_task.py --task bugfix_metric_zero_domain_error --transport gym
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from shutil import rmtree
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.trace import Timer, build_row, write_jsonl  # noqa: E402
from benchmark.tools.task_loader import Task, iter_tasks, load_task, materialise  # noqa: E402
from envs.coding_env.session import EpisodeSession, episode_metadata  # noqa: E402
from envs.coding_env.verifier import VerifyResult  # noqa: E402


def gold_writes(task: Task) -> list[tuple[str, str]]:
    """The gold patch, re-expressed as `write_file` calls the agent could have made.

    Materialising the patched task and diffing file-by-file, rather than parsing the diff,
    keeps this honest: whatever the gold patch really does to the tree is what gets written.
    """
    tmp = Path(tempfile.mkdtemp(prefix=f"gold_{task.id}_"))
    try:
        patched = materialise(task, tmp / "gold", include_tests=False, include_smoke=False, apply_gold=True)
        writes = []
        for path in sorted(patched.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(patched).as_posix()
            original = task.path / rel
            new = path.read_text(encoding="utf-8")
            if not original.is_file() or original.read_text(encoding="utf-8") != new:
                writes.append((rel, new))
        return writes
    finally:
        rmtree(tmp, ignore_errors=True)


def scripted_episode(task: Task) -> list[tuple[str, dict[str, Any]]]:
    """The fixed sequence of tool calls. Identical for both transports."""
    hint = (task.data.get("repo", {}) or {}).get("entrypoint_hint")
    calls: list[tuple[str, dict[str, Any]]] = [("list_files", {"path": "."})]
    if hint:
        calls.append(("read_file", {"path": f"repo/{hint}"}))
    calls += [("write_file", {"path": rel, "content": text}) for rel, text in gold_writes(task)]
    calls += [("run_tests", {}), ("submit", {})]
    return calls


def run_direct(task: Task) -> dict[str, Any]:
    session = EpisodeSession.open(task)
    timer = Timer.start()
    try:
        transcript = []
        for name, args in scripted_episode(task):
            out = session.tool(name, args)
            transcript.append({"tool": name, "ok": out["ok"], "output": out["output"][:200]})
        result = session.grade()
        return {
            "transcript": transcript,
            "result": result.to_dict(),
            "trace": _trace(task, result, session.sandbox.toolbox.log, timer, "direct"),
        }
    finally:
        session.close()


def run_gym(task: Task) -> dict[str, Any]:
    """Same episode, issued as HTTP POSTs against the real FastAPI app.

    # Pattern adapted from: scripts/check_nemo_gym_install.py (B0), itself following
    # nemo-gym/resources_servers/example_session_state_mgmt/tests
    """
    try:
        from unittest.mock import MagicMock

        from nemo_gym.server_utils import ServerClient
        from starlette.testclient import TestClient

        from envs.coding_env.app import CodingResourcesServer, CodingResourcesServerConfig
    except ImportError as e:
        raise SystemExit(f"--transport gym needs the nemo-gym stack: {e}\nTry: uv sync --extra gym") from e

    server = CodingResourcesServer(
        config=CodingResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="", name="coding_env_smoke"),
        server_client=MagicMock(spec=ServerClient),
    )
    row = {"task_id": task.id, "split": task.split, "metadata": episode_metadata(task)}
    timer = Timer.start()
    with TestClient(server.setup_webserver()) as client:
        seed = client.post("/seed_session", json=row)
        seed.raise_for_status()
        transcript = []
        for name, args in scripted_episode(task):
            reply = client.post(f"/{name}", json=args)
            reply.raise_for_status()
            body = reply.json()
            transcript.append({"tool": name, "ok": body["ok"], "output": body["output"][:200]})
        # Grab the transcript before /verify, which closes the session and drops the sandbox.
        tool_log = list(next(iter(server.store.sessions.values())).sandbox.toolbox.log)
        verified = client.post(
            "/verify",
            json={
                **row,
                "responses_create_params": {"input": [{"role": "user", "content": task.data["instruction"]}]},
                # The minimal object that satisfies NeMoGymResponse validation; the scripted
                # episode has no model trajectory, and /verify scores sandbox state anyway.
                "response": {
                    "id": "resp_smoke",
                    "created_at": 0,
                    "model": "smoke",
                    "object": "response",
                    "output": [],
                    "parallel_tool_calls": False,
                    "tool_choice": "auto",
                    "tools": [],
                },
            },
        )
        verified.raise_for_status()
        scored = verified.json()
        return {
            "transcript": transcript,
            "result": scored,
            "live_sessions": len(server.store.sessions),
            "trace": _trace(task, _result_from_json(scored), tool_log, timer, "gym"),
        }


def _result_from_json(scored: dict[str, Any]) -> VerifyResult:
    """Rebuild the structured result from the HTTP response.

    The trace must be produced from what `/verify` actually returned, not from a local
    re-run: if the wire format ever loses a field, the trace should break here rather than
    quietly fall back to a second, unrelated measurement.
    """
    return VerifyResult(
        task_id=scored["task_id"],
        reward=scored["reward"],
        pass_fraction=scored["pass_fraction"],
        passed=scored["tests_passed"],
        collected=scored["tests_total"],
        per_test=scored["per_test"],
        gates=scored["gates"],
        submitted=scored["submitted"],
        tool_calls=scored["tool_calls"],
        limits_enforced=scored["limits_enforced"],
    )


def _trace(task: Task, result: VerifyResult, tool_log: list[Any], timer: Timer, transport: str) -> dict[str, Any]:
    return build_row(
        run_id=f"b3_smoke_{transport}",
        rollout_id=f"{task.id}#0",
        task=task,
        result=result,
        tool_calls=tool_log,
        timer=timer,
        transport=transport,
        agent_kind="scripted_gold",
        model=None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", help="Task id. Default: the first train task.")
    parser.add_argument("--limit", type=int, default=1, help="How many tasks to run. Capped at 3 (spec 1.4).")
    parser.add_argument("--transport", choices=("direct", "gym"), default="direct")
    parser.add_argument("--json", type=Path, help="Write the full report here.")
    parser.add_argument("--trace", type=Path, help="Write rollout traces here as JSONL (analysis/trace_schema.json).")
    args = parser.parse_args()

    if args.task:
        tasks = [load_task(next(p for p in (REPO_ROOT / "benchmark" / "tasks").glob(f"*/{args.task}")))]
    else:
        tasks = list(iter_tasks(("train",)))[: min(args.limit, 3)]

    runner = run_direct if args.transport == "direct" else run_gym
    report = {"transport": args.transport, "tasks": {}}
    ok = True
    for task in tasks:
        outcome = runner(task)
        report["tasks"][task.id] = outcome
        result = outcome["result"]
        gates = result.get("gates") or []
        status = "PASS" if result["reward"] == 1.0 else "FAIL"
        ok &= result["reward"] == 1.0
        print(
            f"{status}  {task.id}  reward={result['reward']:.2f}  "
            f"tests={result.get('tests_passed', result.get('passed'))}/"
            f"{result.get('tests_total', result.get('collected'))}  "
            f"gates={gates or '-'}"
        )

    if args.trace:
        write_jsonl([outcome["trace"] for outcome in report["tasks"].values()], args.trace)
        print(f"wrote {args.trace}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    print("\nThe gold solution scores 1.0 through the real tools." if ok else "\nSMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
