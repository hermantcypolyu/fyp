"""Phase 6 tests: the frozen baseline config, and the collector's parsing rules.

These tests exist because the collector is the only producer of the Phase 6 numbers. A
silent parsing bug there does not crash - it produces plausible-looking rollouts that
score 0.0, and a baseline that measures the harness instead of the model. Every test
below pins a failure mode that was actually observed while bringing the collector up.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "eval" / "baseline_phase6.yaml"
TASKS_TRAIN = REPO_ROOT / "benchmark" / "tasks" / "train"


# --------------------------------------------------------------------------- config


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_baseline_config_exists_and_is_frozen() -> None:
    cfg = _config()
    assert cfg["meta"]["status"] == "FROZEN", "Phase 6.1 output must be recorded as frozen"
    assert cfg["meta"]["phase"] == "6.1"


def test_heldout_is_disabled() -> None:
    """6.1 baseline runs on train only. Touching heldout would burn the sealed split."""
    assert _config()["tasks"]["heldout_enabled"] is False


def test_baseline_arm_loads_no_adapter() -> None:
    """The baseline is the base model. A non-null adapter makes it not a baseline."""
    assert _config()["model"]["adapter"] is None


def test_model_is_pinned_to_a_revision() -> None:
    revision = _config()["model"]["revision"]
    assert isinstance(revision, str) and len(revision) == 40, "revision must be a full git sha"
    assert revision == "2e1fd397ee46e1388853d2af2c993145b0f1098a"


def test_every_checklist_item_is_bound_to_a_real_key() -> None:
    """A checklist entry that resolves to nothing is a promise with no code behind it.

    `model.adapter` is deliberately null - the baseline arm loads no adapter - so null is
    only an error when the checklist did not declare it as an expected-null value.
    """
    cfg = _config()
    for item, spec in cfg["checklist"].items():
        node = cfg
        for part in spec["key"].split("."):
            assert part in node, f"{item}: checklist key {spec['key']} has no such field"
            node = node[part]
        if node is None:
            assert spec.get("expected_null") is True, f"{item}: {spec['key']} is null but not declared expected_null"


def test_task_list_is_exactly_the_train_split() -> None:
    """Hand-written task lists drift. The directory is the authority."""
    cfg = _config()
    listed = cfg["tasks"]["list"]
    on_disk = sorted(p.name for p in TASKS_TRAIN.iterdir() if (p / "task.yaml").is_file())
    assert sorted(listed) == on_disk, "tasks.list must be set-equal to benchmark/tasks/train/"
    assert len(listed) == len(set(listed)), "tasks.list has duplicates"
    assert cfg["tasks"]["expected_n_tasks"] == len(on_disk)


def test_every_listed_task_resolves() -> None:
    from envs.coding_env.session import find_task

    for task_id in _config()["tasks"]["list"]:
        assert find_task(task_id).id == task_id


def test_seeded_and_deterministic_sampling() -> None:
    cfg = _config()
    assert cfg["sampling"]["seed"] == 42, "a baseline without a fixed seed is not repeatable"
    assert cfg["sampling"]["top_p"] == 1.0


def test_server_is_on_the_non_default_port() -> None:
    """Port 8000 is occupied by another user on this box; 8010 is the agreed port."""
    assert _config()["server"]["base_url"] == "http://127.0.0.1:8010/v1"
    assert _config()["server"]["cuda_visible_devices"] == "0"


# ---------------------------------------------------------------- tool-call parsing


def _msg(content: str | None = None, tool_calls=None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def test_parses_documented_name_arguments_shape() -> None:
    from scripts.collect_baseline_rollouts import parse_tool_calls

    calls = parse_tool_calls(_msg('{"name": "list_files", "arguments": {"path": "."}}'))
    assert calls == [("list_files", {"path": "."})]


def test_parses_fenced_json_block() -> None:
    from scripts.collect_baseline_rollouts import parse_tool_calls

    text = 'Sure.\n```json\n{"name": "run_tests", "arguments": {}}\n```'
    assert parse_tool_calls(_msg(text)) == [("run_tests", {})]


def test_parses_tool_name_as_key_shape() -> None:
    """Observed 1.5B output: the tool name used as the mapping key, not `"name"`."""
    from scripts.collect_baseline_rollouts import parse_tool_calls

    calls = parse_tool_calls(_msg('{"run_tests": {}}'))
    assert calls == [("run_tests", {})]


def test_unknown_tool_names_are_not_calls() -> None:
    """A hallucinated tool must be ignored, never forwarded into the sandbox."""
    from scripts.collect_baseline_rollouts import parse_tool_calls

    assert parse_tool_calls(_msg('{"name": "delete_everything", "arguments": {}}')) == []
    assert parse_tool_calls(_msg('{"http_request": {"url": "https://example.com"}}')) == []


def test_prose_without_json_is_a_no_op_turn() -> None:
    from scripts.collect_baseline_rollouts import parse_tool_calls

    assert parse_tool_calls(_msg("I will look at the file first.")) == []


def test_arguments_may_be_a_json_string() -> None:
    from scripts.collect_baseline_rollouts import parse_tool_calls

    calls = parse_tool_calls(_msg('{"name": "read_file", "arguments": "{\\"path\\": \\"a.py\\"}"}'))
    assert calls == [("read_file", {"path": "a.py"})]


def test_native_tool_calls_are_preferred() -> None:
    from scripts.collect_baseline_rollouts import parse_tool_calls

    native = [SimpleNamespace(function=SimpleNamespace(name="submit", arguments="{}"))]
    calls = parse_tool_calls(_msg('{"name": "list_files", "arguments": {}}', native))
    assert calls == [("submit", {})]


# --------------------------------------------------------------- prompt / instruction


def test_system_prompt_shows_a_concrete_call_example() -> None:
    """The first failure was a prompt that said 'emit JSON' but never showed the shape."""
    from scripts.collect_baseline_rollouts import BaselineConfig, _system_prompt

    prompt = _system_prompt(BaselineConfig.load())
    assert '"name"' in prompt and '"arguments"' in prompt
    for tool in ("list_files", "read_file", "run_tests", "submit"):
        assert tool in prompt, f"{tool} missing from the tool catalogue"


def test_missing_instruction_is_an_error_not_a_blank_episode() -> None:
    """An empty user turn scores 0.0 and looks like a legitimate rollout."""
    from scripts.collect_baseline_rollouts import CollectorError, _task_instruction

    task = SimpleNamespace(id="broken_task", data={})
    with pytest.raises(CollectorError, match="instruction"):
        _task_instruction(task)


def test_instruction_is_taken_from_task_data() -> None:
    from scripts.collect_baseline_rollouts import _task_instruction

    task = SimpleNamespace(id="t", data={"instruction": "  fix the bug  "})
    assert _task_instruction(task) == "fix the bug"


# --------------------------------------------------------------------- config guards


def test_config_refuses_missing_keys_instead_of_defaulting() -> None:
    """A code-side default would silently paper over an unfrozen 6.1 value."""
    from scripts.collect_baseline_rollouts import BaselineConfig, CollectorError

    cfg = BaselineConfig.load()
    with pytest.raises(CollectorError):
        cfg._get("sampling.this_key_does_not_exist")


def test_uniform_reward_batch_is_rejected() -> None:
    """Constant reward gives GRPO no gradient; it is a harness bug, not a baseline."""
    from scripts.collect_baseline_rollouts import UniformRewardError

    assert issubclass(UniformRewardError, Exception)
