#!/usr/bin/env python
"""Prove that a *built-in, unmodified* NeMo Gym environment runs on this machine.

This is the AGENT_BUILD_SPEC B0 hard gate ("one built-in environment inspected and
run unmodified"). It imports two shipped environments straight out of a NeMo Gym
checkout and drives their real FastAPI apps - `setup_webserver()`, the session
middleware, the tool routes and `POST /verify` - through Starlette's TestClient.

It deliberately does NOT go through `gym env start`, because that path boots a Ray
cluster, and Ray's raylet does not start on native Windows (see
docs/nemo_gym_notes.md, "What we could not run here"). Everything *inside* the
environment - config model, server construction, routing, per-rollout session
state, verifier, reward - is exercised for real.

Nothing in the NeMo Gym checkout is modified. Usage:

    uv run --extra gym python scripts/check_nemo_gym_install.py --gym-path ../_vendor/nemo-gym

On Windows you additionally need the POSIX `resource` shim on PYTHONPATH; see
docs/nemo_gym_notes.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_gym(gym_path: Path) -> None:
    if not (gym_path / "nemo_gym").is_dir():
        raise SystemExit(
            f"No NeMo Gym checkout at {gym_path}.\n"
            "Clone it first:  git clone https://github.com/NVIDIA-NeMo/Gym.git\n"
            "then pass --gym-path, or set NEMO_GYM_PATH."
        )
    sys.path.insert(0, str(gym_path))


def _single_tool_call(client_factory) -> dict[str, Any]:
    """resources_servers/example_single_tool_call - stateless single tool + trivial verifier."""
    from nemo_gym.server_utils import ServerClient
    from resources_servers.example_single_tool_call.app import (
        SimpleWeatherResourcesServer,
        SimpleWeatherResourcesServerConfig,
    )

    server = SimpleWeatherResourcesServer(
        config=SimpleWeatherResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="", name="demo_weather"),
        server_client=MagicMock(spec=ServerClient),
    )
    with client_factory(server.setup_webserver()) as client:
        tool = client.post("/get_weather", json={"city": "Hong Kong"})
        verify = client.post(
            "/verify",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "weather?"}]},
                "response": _empty_response(),
            },
        )
    return {
        "tool_status": tool.status_code,
        "tool_body": tool.json(),
        "verify_status": verify.status_code,
        "reward": verify.json().get("reward"),
    }


def _session_state(client_factory) -> dict[str, Any]:
    """resources_servers/example_session_state_mgmt - the pattern our sandbox uses.

    Per-rollout state is keyed by `request.session[SESSION_ID_KEY]`, seeded in
    /seed_session and read back in /verify.
    """
    from nemo_gym.server_utils import ServerClient
    from resources_servers.example_session_state_mgmt.app import (
        StatefulCounterResourcesServer,
        StatefulCounterResourcesServerConfig,
    )

    server = StatefulCounterResourcesServer(
        config=StatefulCounterResourcesServerConfig(host="127.0.0.1", port=0, entrypoint="", name="demo_counter"),
        server_client=MagicMock(spec=ServerClient),
    )
    with client_factory(server.setup_webserver()) as client:
        seed = client.post("/seed_session", json={"initial_count": 10})
        client.post("/increment_counter", json={"count": 5})
        client.post("/increment_counter", json={"count": 7})
        value = client.post("/get_counter_value", json={})
        verify = client.post(
            "/verify",
            json={
                "responses_create_params": {"input": [{"role": "user", "content": "count to 22"}]},
                "response": _empty_response(),
                "expected_count": 22,
            },
        )
    return {
        "seed_status": seed.status_code,
        "counter_after_two_increments": value.json().get("count"),
        "verify_status": verify.status_code,
        "reward": verify.json().get("reward"),
        "session_keys_held_by_server": len(server.session_id_to_counter),
    }


def _empty_response() -> dict[str, Any]:
    """The minimal object that satisfies NeMoGymResponse validation in /verify."""
    return {
        "id": "resp_demo",
        "created_at": 0,
        "model": "demo",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gym-path",
        default=os.environ.get("NEMO_GYM_PATH", str(REPO_ROOT.parent / "_vendor" / "nemo-gym")),
        help="Path to a NeMo Gym checkout.",
    )
    parser.add_argument("--out", default=None, help="Write the JSON report here.")
    args = parser.parse_args()

    gym_path = Path(args.gym_path).expanduser().resolve()
    _load_gym(gym_path)

    from starlette.testclient import TestClient

    report: dict[str, Any] = {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "gym_path": str(gym_path),
        "gym_commit": _git_head(gym_path),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "environments": {},
    }
    report["environments"]["example_single_tool_call"] = _single_tool_call(TestClient)
    report["environments"]["example_session_state_mgmt"] = _session_state(TestClient)

    ok = (
        report["environments"]["example_single_tool_call"]["reward"] == 1.0
        and report["environments"]["example_session_state_mgmt"]["reward"] == 1.0
        and report["environments"]["example_session_state_mgmt"]["counter_after_two_increments"] == 22
    )
    report["all_checks_passed"] = ok

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {out}", file=sys.stderr)
    return 0 if ok else 1


def _git_head(path: Path) -> str:
    head = path / ".git" / "HEAD"
    if not head.is_file():
        return "unknown"
    ref = head.read_text(encoding="utf-8").strip()
    if ref.startswith("ref: "):
        target = path / ".git" / ref[5:]
        return target.read_text(encoding="utf-8").strip() if target.is_file() else "unknown"
    return ref


if __name__ == "__main__":
    raise SystemExit(main())
