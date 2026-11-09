"""Phase 6K: the prompt states the required working procedure.

Phase 6J removed the navigation defect - 20 of 20 failing calls became 16 of 16 correct
paths - and the reward did not move. The residual failure was different: both rollouts
read the file and then stopped. Neither edited, neither ran the tests, neither submitted,
so the verifier had nothing to score.

The Phase 6K remediation is again prompt-only and again deliberately narrow. It names the
four phases and says that reading alone is not progress. These tests pin the two
properties that keep the comparison meaningful:

1. the prompt requires the four phases by name;
2. the guidance leaks nothing - no correct code change, no hidden test, no reference
   solution, no mention of the actual bug - and the harness still performs no edit, no
   test run, and no submission on the model's behalf.

Test 5 is the important one. The whole point of the experiment is to measure whether the
model can be moved from reading to doing by instruction alone. If the harness started
editing files or running tests for it, the experiment would measure the harness instead.
"""

from __future__ import annotations

import pytest

from scripts.collect_baseline_rollouts import (
    WORKFLOW_GUIDANCE,
    BaselineConfig,
    _system_prompt,
)


@pytest.fixture(scope="module")
def prompt() -> str:
    """The system prompt exactly as the collector builds it."""
    return _system_prompt(BaselineConfig.load())


# --------------------------------------------------- 1. the four phases are named
@pytest.mark.parametrize("phase", ["read", "edit", "run_tests", "submit"])
def test_prompt_names_each_required_phase(prompt: str, phase: str) -> None:
    """Each phase appears by the name of the tool that performs it."""
    assert phase in WORKFLOW_GUIDANCE
    assert phase in prompt


def test_prompt_states_the_phases_are_ordered(prompt: str) -> None:
    """Order matters: the phases are a sequence, not a menu of options."""
    assert "in order" in prompt


def test_prompt_forbids_stopping_after_reading(prompt: str) -> None:
    """The observed 6J failure is named and ruled out without naming the bug."""
    assert "Reading is not progress on its own." in prompt


# ------------------------------------------------------------ 2. nothing leaked
@pytest.mark.parametrize(
    "leak",
    [
        "next_page",
        "zfill",
        "ljust",
        "off-by-one",
        "off by one",
        "ordinal",
        "rollover",
        "hidden test",
        "gold patch",
        "reference solution",
        "the bug is",
        "the fix is",
    ],
)
def test_guidance_reveals_no_solution(leak: str) -> None:
    """The guidance says how to work, never what the answer is."""
    assert leak.lower() not in WORKFLOW_GUIDANCE.lower()


def test_guidance_does_not_show_the_expected_code(prompt: str) -> None:
    """No code block, so no candidate patch can be lifted from the prompt."""
    assert "```" not in WORKFLOW_GUIDANCE
    assert "def " not in WORKFLOW_GUIDANCE


def test_guidance_offers_no_automatic_action() -> None:
    """The harness must not promise help it performs itself."""
    lowered = WORKFLOW_GUIDANCE.lower()
    for promise in ("will be applied", "will be run for you", "automatically"):
        assert promise not in lowered


# ----------------------------------------------- 3. the guidance is actually sent
def test_guidance_is_included_in_the_system_prompt(prompt: str) -> None:
    """A constant nothing appends is a comment, not a prompt change."""
    assert WORKFLOW_GUIDANCE in prompt


def test_guidance_sits_alongside_the_path_guidance(prompt: str) -> None:
    """The 6J remediation is still present; 6K adds to it rather than replacing it."""
    assert "The repository root is `repo/`." in prompt
    assert prompt.index("The repository root is `repo/`.") < prompt.index(WORKFLOW_GUIDANCE)


def test_guidance_precedes_the_tool_catalogue(prompt: str) -> None:
    """The procedure is read before the list of tools it refers to."""
    assert prompt.index(WORKFLOW_GUIDANCE) < prompt.index("Available tools:")
