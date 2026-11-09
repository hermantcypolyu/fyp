"""Phase 6J: the prompt states the workspace root and the required path prefix.

Phase 6I produced four infrastructure-valid rollouts that all scored -1.0. The cause was
not sampling, the verifier, or the model: every rollout addressed files as
``src/pagination.py`` while the sandbox only resolves paths under ``repo/``. Twenty of
twenty tool calls returned "no such file", and two rollouts re-issued byte-identical
failing calls until ``max_turns``.

The Phase 6J remediation is prompt-only. These tests pin the two properties that make the
comparison meaningful:

1. the prompt now states the root, the prefix, and what to do after a miss;
2. nothing else changed - no answer leak, no hidden tests, and the sandbox still rejects
   unprefixed paths instead of quietly repairing them.

Test 6 is the important one. The temptation when a model keeps missing a path is to
prepend ``repo/`` inside the tool. That would make the tool an accomplice to the bug and
would remove the behaviour this experiment is meant to measure, so the strictness is
asserted rather than assumed.
"""

from __future__ import annotations

import pytest

from scripts.collect_baseline_rollouts import BaselineConfig, _system_prompt


@pytest.fixture(scope="module")
def prompt() -> str:
    """The system prompt exactly as the collector builds it."""
    return _system_prompt(BaselineConfig.load())


# ------------------------------------------------------- 1. root and prefix stated
def test_prompt_states_the_repository_root_is_repo(prompt: str) -> None:
    """The root has to be named, not inferred from a tool description."""
    assert "The repository root is `repo/`." in prompt


def test_prompt_requires_paths_to_begin_with_repo(prompt: str) -> None:
    assert "All file paths passed to repository tools must begin with `repo/`." in prompt


# --------------------------------------------- 2. a correct path appears as example
def test_prompt_shows_a_correctly_prefixed_path(prompt: str) -> None:
    """The worked example must model the shape the sandbox accepts."""
    assert "repo/src/pagination.py" in prompt


def test_worked_example_does_not_teach_an_unprefixed_path(prompt: str) -> None:
    """The old example used ``src/app.py``, which the sandbox rejects.

    This is the specific regression that produced Phase 6I's 20 not-found calls: the
    prompt demonstrated the exact format that fails.
    """
    assert '"path": "src/app.py"' not in prompt
    assert '"path": "src/pagination.py"' not in prompt


def test_prompt_shows_the_visible_repository_tree(prompt: str) -> None:
    assert "repo/" in prompt
    assert "├── src/" in prompt
    assert "└── tests/" in prompt


# --------------------------------- 3. instructs against repeating an identical call
def test_prompt_tells_the_model_not_to_repeat_an_identical_failed_call(prompt: str) -> None:
    """Two Phase 6I rollouts burned the whole budget on one repeated call."""
    assert "do not repeat the identical tool call" in prompt


# ------------------------------------------------------- 4/5. no answer or test leak
@pytest.mark.parametrize(
    "leak",
    [
        "off-by-one",
        "off by one",
        "ceiling",
        "-(-n",
        "floor division",
        "trailing partial page counts",
        "reports exactly one page",
    ],
)
def test_prompt_does_not_reveal_the_fix(prompt: str, leak: str) -> None:
    """Navigation guidance only: the diagnosis stays in the task instruction."""
    assert leak.lower() not in prompt.lower()


@pytest.mark.parametrize("leak", ["hidden test", "gold patch", "reference solution", "solution/"])
def test_prompt_does_not_expose_hidden_tests_or_solutions(prompt: str, leak: str) -> None:
    """The prompt may mention the visible `tests/` directory; nothing more.

    `run_tests`' own description already explains that the smoke tests are not the
    grading tests, so that phrase is expected and is not a leak.
    """
    assert leak.lower() not in prompt.lower()


def test_prompt_keeps_the_instruction_as_the_only_bug_description(prompt: str) -> None:
    """The remediation must not restate the task in its own words."""
    cfg = BaselineConfig.load()
    assert prompt.startswith(cfg.system_message), "base system message must be unchanged"
    assert "pagination" not in cfg.system_message.lower()


# ---------------------------------------- 6. tools still reject, never auto-repair
def test_sandbox_rejects_an_unprefixed_path_instead_of_repairing_it(tmp_path) -> None:
    """``src/pagination.py`` must not be silently rewritten to ``repo/src/pagination.py``.

    The remediation is a prompt change. If the tool auto-corrected, a Phase 6J
    improvement would measure the harness, not the prompt, and Phase 6I's diagnosis
    would be untestable.
    """
    from envs.coding_env.tools import Toolbox

    (tmp_path / "repo" / "src").mkdir(parents=True)
    (tmp_path / "repo" / "src" / "pagination.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "smoke").mkdir()

    box = Toolbox(root=tmp_path)
    result = box.read_file("src/pagination.py")

    assert result.ok is False
    # The file exists at repo/src/pagination.py; an auto-repair would have found it.
    assert "no such file" in result.output.lower()


def test_sandbox_accepts_the_prefixed_path_the_prompt_teaches(tmp_path) -> None:
    """The path the prompt now teaches must actually work."""
    from envs.coding_env.tools import Toolbox

    (tmp_path / "repo" / "src").mkdir(parents=True)
    (tmp_path / "repo" / "src" / "pagination.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "smoke").mkdir()

    box = Toolbox(root=tmp_path)
    result = box.read_file("repo/src/pagination.py")

    assert result.ok is True


# ------------------------------------- 7. everything except the prompt is unchanged
def test_nothing_but_the_prompt_changed_in_this_experiment() -> None:
    """Phase 6J changes exactly one lever.

    Model, task, seeds, sampling, budgets and reward must equal the Phase 6I control,
    otherwise the A/B is confounded and the comparison means nothing.

    The single task under test is selected by the run overlay, not the base config, so
    this test pins only what the base config owns.
    """
    cfg = BaselineConfig.load()

    assert cfg.model_name == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert cfg.model_revision == "2e1fd397ee46e1388853d2af2c993145b0f1098a"
    assert cfg.prompt_version == "baseline_phase6_v1"
    assert cfg.temperature == pytest.approx(1.0)
    assert cfg.top_p == pytest.approx(1.0)
    assert cfg.seed == 42
    assert cfg.seed_derivation == "sha256_u32_v1"
    # The held-out split stays sealed: Phase 6J is a train-split diagnostic, not the
    # one permitted final evaluation.
    assert cfg.raw["tasks"]["heldout_enabled"] is False
    assert cfg.raw["tasks"]["split"] == "train"
    # The guidance is appended by `_system_prompt`; the configured message is untouched.
    assert cfg.system_message.count("repo/") == 0
