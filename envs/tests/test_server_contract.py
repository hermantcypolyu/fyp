"""Static contract for the two modules the default dev environment cannot import.

`nemo_gym/server_utils.py` has a top-level `import resource`, so `envs/coding_env/app.py`
and `agent.py` need either POSIX or B0's `_vendor/win_compat/` shim - which lives outside
this repo and is not installed by `uv sync`. They *have* been run with the shim
(`scripts/smoke_one_task.py --transport gym`, reward 1.0 over the real FastAPI app), but
that cannot be part of the default `make test`.

So their structure is asserted with `ast` instead: the seven routes exist and are named
exactly as the tool schema says, the verify path goes through the one shared verifier, and
the agent really is a thin subclass rather than a reimplemented loop. That catches the
failures that actually happen - a typo'd route, a second scoring path, a copy-pasted loop -
in an environment where importing is not an option.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from envs.coding_env.tools import TOOL_DEFINITIONS

ENV_DIR = Path(__file__).resolve().parents[1] / "coding_env"
TOOL_NAMES = [t["name"] for t in TOOL_DEFINITIONS]


def _tree(name: str) -> ast.Module:
    return ast.parse((ENV_DIR / name).read_text(encoding="utf-8"))


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name} not found")


def _route_paths(tree: ast.Module) -> set[str]:
    """Every `app.post("/x")(...)` literal in the module."""
    paths = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "post"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            paths.add(node.args[0].value)
    return paths


# --------------------------------------------------------------------------- app.py


def test_app_exposes_exactly_the_seven_tools():
    assert _route_paths(_tree("app.py")) == {f"/{n}" for n in TOOL_NAMES}


def test_every_route_has_a_handler_of_the_same_name():
    """A route whose path and method disagree fails only at rollout time, in someone else's log."""
    methods = {
        n.name for n in _class(_tree("app.py"), "CodingResourcesServer").body if isinstance(n, ast.AsyncFunctionDef)
    }
    assert set(TOOL_NAMES) <= methods


def test_verify_and_seed_session_are_overridden():
    methods = {
        n.name for n in _class(_tree("app.py"), "CodingResourcesServer").body if isinstance(n, ast.AsyncFunctionDef)
    }
    assert {"verify", "seed_session"} <= methods


def _imported_modules(tree: ast.Module) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_verify_goes_through_the_one_shared_verifier():
    """5.6: `/verify` must reach `verifier.verify` via the session, not open a second path."""
    tree = _tree("app.py")
    assert "session.grade()" in (ENV_DIR / "app.py").read_text(encoding="utf-8")
    ours = {m for m in _imported_modules(tree) if m.startswith(("envs.", "benchmark."))}
    assert ours == {"envs.coding_env.session"}, f"app.py must reach the verifier only via session: {ours}"


def test_verify_response_carries_the_reward_floor():
    """ADR 0006: reward is meaningless without `floor_tests`, so the wire format must carry it.

    The gym transport rebuilds `VerifyResult` from this response
    (`smoke_one_task._result_from_json`). If the field were dropped, the reconstructed row
    would silently read `floor_tests = 0` and every gym-side reward would be the raw pass
    fraction again - the exact regression ADR 0006 exists to prevent, on one transport only.
    """
    response = _class(_tree("app.py"), "CodingVerifyResponse")
    annotated = {n.target.id for n in response.body if isinstance(n, ast.AnnAssign)}
    assert {"floor_tests", "tests_passed", "tests_total", "pass_fraction"} <= annotated

    populated = (ENV_DIR / "app.py").read_text(encoding="utf-8")
    assert "floor_tests=result.floor_tests" in populated


def test_app_is_the_only_place_the_environment_touches_nemo_gym():
    offenders = []
    for path in ENV_DIR.rglob("*.py"):
        if path.name in {"app.py", "agent.py"}:
            continue
        if any(
            m.startswith(("nemo_gym", "responses_api_agents"))
            for m in _imported_modules(ast.parse(path.read_text(encoding="utf-8")))
        ):
            offenders.append(path.name)
    assert offenders == [], f"nemo_gym leaked into {offenders}; keep the core framework-free"


# --------------------------------------------------------------------------- agent.py


def test_agent_is_a_subclass_of_the_builtin():
    bases = [ast.unparse(b) for b in _class(_tree("agent.py"), "CodingAgent").bases]
    assert bases == ["SimpleAgent"]


def test_agent_overrides_one_method_and_writes_no_loop():
    """Requirement: a thin subclass, not a reimplemented harness."""
    cls = _class(_tree("agent.py"), "CodingAgent")
    overrides = [n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert overrides == ["responses"]
    assert not any(isinstance(n, (ast.While, ast.For)) for n in ast.walk(cls))
    assert "SimpleAgent.responses(" in (ENV_DIR / "agent.py").read_text(encoding="utf-8")


def test_agent_reads_the_per_task_turn_cap():
    source = (ENV_DIR / "agent.py").read_text(encoding="utf-8")
    assert "max_turns" in source
    assert "max_steps" in source


# --------------------------------------------------------------------------- config


def test_config_points_at_our_modules():
    config = (ENV_DIR / "configs" / "coding_env.yaml").read_text(encoding="utf-8")
    assert "entrypoint: app.py" in config
    assert "entrypoint: agent.py" in config


@pytest.mark.gym
def test_modules_import_when_the_gym_stack_is_present():
    """Skipped unless the whole stack is installed. The real check; the rest stands in for it."""
    pytest.importorskip("nemo_gym")
    pytest.importorskip("fastapi")  # a stray PYTHONPATH can expose one without the other
    import envs.coding_env.agent  # noqa: F401
    import envs.coding_env.app  # noqa: F401
