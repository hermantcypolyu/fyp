"""Documentation-consistency checks for the Phase 6 status language.

The Phase 6 bring-up batch failed baseline acceptance. The risk this module guards
against is not a crash: it is a document that quietly reads as if the batch *was*
accepted, or a metric restated without its denominator. Those claims are cheap to
verify mechanically, so they are verified here rather than by re-reading prose.

Every assertion below is grounded in the preserved 48-row evidence in
`results/raw/phase6_baseline_1p5b/`. Tests that need that directory skip when it is
absent, because it is gitignored and a fresh clone legitimately does not have it.
The pure-text checks on the tracking document run everywhere.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STATUS_DOC = REPO_ROOT / "docs" / "CURRENT_PROJECT_STATUS.md"
RUN_DIR = REPO_ROOT / "results" / "raw" / "phase6_baseline_1p5b"
ROWS = RUN_DIR / "rollouts.jsonl"


def _status_doc() -> str:
    assert STATUS_DOC.is_file(), f"missing tracking document: {STATUS_DOC}"
    return STATUS_DOC.read_text(encoding="utf-8")


def _phase6_section() -> str:
    """The Phase 6 portion of the tracking document, without the later phases."""
    text = _status_doc()
    start = text.index("### Phase 6 ")
    end = text.index("### Phase 7 ")
    return text[start:end]


def _flat(text: str) -> str:
    """Flatten prose so claims can be checked independently of markdown layout.

    Two formatting artefacts would otherwise make a present claim look absent:
    hard-wrapping at the line limit, and emphasis or blockquote markers sitting
    between words (`they are **not** reward ...`, `used as > the accepted`).
    Both are stripped here so the assertions target wording, not rendering.
    """
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r"(?m)^\s*>+\s?", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.replace(" > ", " ")


def _rows() -> list[dict]:
    if not ROWS.is_file():
        pytest.skip(f"preserved Phase 6 evidence not present: {ROWS}")
    return [json.loads(line) for line in ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]


def _reward_eligible(rows: list[dict]) -> list[dict]:
    """Mirror of baseline_analysis.reward_eligible_rows without importing the script."""
    out = []
    for row in rows:
        failure = row.get("infrastructure_failure")
        if row.get("termination_reason") == "infrastructure_error":
            continue
        if isinstance(failure, dict) and failure.get("reward_eligible") is False:
            continue
        # The frozen rows predate the explicit field: a 400 context overflow is the only
        # error class present in this batch, and it carries no scorable result.
        message = str((row.get("error") or {}).get("message") or "")
        if "maximum context length" in message and "400" in message:
            continue
        out.append(row)
    return out


# ------------------------------------------------------------------ status language


def test_phase6_status_does_not_claim_completion_of_6_2_or_6_3() -> None:
    """The rejected batch must not be described as a completed 6.2/6.3 deliverable."""
    section = _phase6_section().lower()
    for forbidden in ("6.2 collected", "6.2 complete", "6.3 complete"):
        assert forbidden not in section, f"Phase 6 still claims {forbidden!r}"


def test_phase6_status_separates_machinery_from_acceptance() -> None:
    """Proven machinery and an accepted baseline are different claims."""
    section = _flat(_phase6_section()).lower()
    assert "collector execution completed but the batch failed acceptance" in section
    assert "diagnostic analysis" in section
    assert "pending recollection" in section


def test_phase6_marks_the_batch_as_diagnostic_evidence_only() -> None:
    section = _flat(_phase6_section()).lower()
    assert "diagnostic evidence only" in section
    assert "must not be used as the accepted baseline" in section
    assert "as grpo training input" in section


def test_phase6_names_the_section_a_diagnostic_analysis_not_an_accepted_baseline() -> None:
    assert "6.3 Diagnostic analysis of the rejected bring-up batch" in _phase6_section()


def test_phase6_does_not_call_the_rejected_batch_a_baseline_analysis() -> None:
    section = _phase6_section()
    assert "**6.3 Baseline analysis**" not in section
    assert "baseline analysis" not in section.lower()


# ------------------------------------------------------------------ checkbox semantics


def test_non_zero_within_task_variance_is_not_marked_complete() -> None:
    """Pooled cross-task spread must not be passed off as within-group variation."""
    section = _phase6_section()
    assert "- [ ] **Non-zero within-task reward variation**" in section


def test_zero_solved_rollouts_is_recorded_rather_than_implied() -> None:
    section = _phase6_section().lower()
    assert "- [ ] **successful trajectories saved**" in section
    assert "0 solved rollouts" in section
    assert "0 positive-reward rollouts" in section


def test_no_checkbox_states_that_successful_trajectories_were_saved() -> None:
    """A submitted-but-unsolved rollout must never be ticked as a success."""
    section = _phase6_section()
    for line in section.splitlines():
        if "successful trajectories saved" in line.lower():
            assert line.lstrip().startswith("- [ ]"), line


def test_infrastructure_rows_are_distinguished_from_model_failures() -> None:
    section = _flat(_phase6_section()).lower()
    assert "infrastructure-invalid" in section
    assert "not reward \u22121.0 model failures" in section
    assert "zero valid rollouts" in section


# ------------------------------------------------------------------ metric definitions


def test_token_usage_is_described_as_cumulative_not_single_request_context() -> None:
    section = _flat(_phase6_section()).lower()
    assert "cumulative across all requests" in section
    assert "not the context length of" in section


def test_runtime_distinguishes_rollout_mean_from_batch_wall_clock() -> None:
    section = _flat(_phase6_section())
    assert "1,168.4" in section, "batch wall-clock duration must be stated separately"
    assert "not the batch duration" in section


def test_pooled_stdev_is_explicitly_not_within_task_variation() -> None:
    section = _flat(_phase6_section()).lower()
    assert "pooled standard deviation reflects differences across tasks" in section
    assert "within-task variation" in section


def test_hidden_test_fraction_is_not_presented_as_pass_at_1() -> None:
    section = _flat(_phase6_section()).lower()
    assert "112/496" in section
    assert "not a task-level success rate" in section
    assert "must not be interpreted as pass@1" in section


def test_valid_submission_ambiguity_is_resolved_into_named_rates() -> None:
    section = _phase6_section()
    assert "Candidate-submission rate" in section
    assert "Solved-submission rate" in section
    assert "| Valid-submission rate" not in section


def test_security_gate_is_scoped_to_the_two_security_gates() -> None:
    section = _phase6_section()
    assert "sandbox_escape" in section
    assert "tests_modified" in section


def test_overflow_explanation_keeps_the_turn_13_correction() -> None:
    section = _flat(_phase6_section())
    assert "turn 13" in section.lower()
    assert "7,990 message tokens" in section
    assert "9,014" in section
    assert "shrinking the initial prompt would not have fixed" in section.lower()


def test_decision_block_states_decision_c_and_baseline_invalid() -> None:
    section = _flat(_phase6_section())
    assert "Decision: C" in section
    assert "`baseline_valid`: false" in section
    assert "GRPO training suitability: failed" in section


def test_no_training_or_kl_claims_were_introduced() -> None:
    """This is a documentation task: it must not imply a model update happened."""
    section = _flat(_phase6_section()).lower()
    assert "model parameter update: not performed" in section
    assert "reward_v2` live integration: not approved" in section
    assert not re.search(r"kl (divergence )?(was|is) (finite|measured)", section)


# ------------------------------------------------------------------ evidence agreement


def test_evidence_has_forty_eligible_and_eight_infrastructure_rows() -> None:
    rows = _rows()
    eligible = _reward_eligible(rows)
    assert len(rows) == 48
    assert len(eligible) == 40
    assert len(rows) - len(eligible) == 8


def test_evidence_has_no_solved_and_no_positive_reward_rollouts() -> None:
    eligible = _reward_eligible(_rows())
    solved = [r for r in eligible if float(r["result"]["reward"]) == 1.0]
    positive = [r for r in eligible if float(r["result"]["reward"]) > 0]
    assert solved == []
    assert positive == []


def test_every_reward_eligible_task_group_has_zero_variance() -> None:
    eligible = _reward_eligible(_rows())
    groups: dict[str, set[float]] = {}
    for row in eligible:
        groups.setdefault(row["task"]["id"], set()).add(float(row["result"]["reward"]))
    assert groups, "expected at least one reward-eligible task group"
    for task_id, values in groups.items():
        assert len(values) == 1, f"{task_id} shows within-task variance {sorted(values)}"


def test_document_agrees_with_the_measured_eligible_and_invalid_counts() -> None:
    section = _phase6_section()
    rows = _rows()
    eligible = len(_reward_eligible(rows))
    assert f"{eligible} reward-eligible" in section or "40 reward-eligible" in section
    assert "8 are infrastructure-invalid" in section or "8 context overflows" in section


def test_context_overflow_rows_are_excluded_from_reward_metrics() -> None:
    """The 8 overflow rows must not contribute a reward to the analysed population."""
    rows = _rows()
    overflow = [r for r in rows if "maximum context length" in str((r.get("error") or {}).get("message") or "")]
    assert len(overflow) == 8
    eligible_ids = {id(r) for r in _reward_eligible(rows)}
    for row in overflow:
        assert id(row) not in eligible_ids


# ------------------------------------------------------------------ category arithmetic
#
# The broad symptom "reached max_turns without submitting for -1.0" spans two mutually
# exclusive primary categories. Reporting the broad count as if it were the primary count
# overstates `no_submission_turns_exhausted`, which is exactly the error corrected here.


def _broad_max_turn_rows() -> list[dict]:
    """Reward-eligible rows that reached the turn budget without submitting at -1.0."""
    out = []
    for row in _reward_eligible(_rows()):
        result = row.get("result") or {}
        if row.get("termination_reason") != "max_turns":
            continue
        if result.get("submitted") is True:
            continue
        if float(result.get("reward", 0.0)) != -1.0:
            continue
        out.append(row)
    return out


def _primary_category(row: dict) -> str:
    """First-match primary label, mirroring baseline_analysis.failure_category order.

    Only the distinctions needed to split the broad symptom are reproduced here: an
    infrastructure row is excluded upstream, a syntax signal outranks the no-submission
    branch, and everything else that did not submit falls into the turn-exhausted label.
    """
    for step in row.get("steps") or []:
        if step.get("ok"):
            continue
        observation = str(step.get("observation_summary") or "")
        if any(
            marker in observation
            for marker in ("SyntaxError", "IndentationError", "TabError", "invalid syntax", "unexpected EOF")
        ):
            return "syntax_failure"
    error_message = str((row.get("error") or {}).get("message") or "")
    if "unexpected EOF" in error_message:
        return "syntax_failure"
    return "no_submission_turns_exhausted"


def test_unsupported_count_34_is_absent_from_phase6_documentation() -> None:
    """34 was never supported by the frozen rows; no Phase 6 document may restate it."""
    assert not re.search(r"\b34\b", _flat(_phase6_section())), "Phase 6 status still contains 34"
    smoke = REPO_ROOT / "docs" / "PHASE6B_SMOKE_PROTOCOL.md"
    if smoke.is_file():
        flat = _flat(smoke.read_text(encoding="utf-8"))
        assert not re.search(r"\b34\b", flat), "smoke protocol still contains 34"


def test_broad_max_turn_symptom_is_thirty_two_rows() -> None:
    assert len(_broad_max_turn_rows()) == 32


def test_broad_symptom_splits_into_twenty_four_and_eight_primary_rows() -> None:
    split = Counter(_primary_category(r) for r in _broad_max_turn_rows())
    assert split["no_submission_turns_exhausted"] == 24
    assert split["syntax_failure"] == 8
    assert sum(split.values()) == 32


def test_documentation_states_the_thirty_two_twenty_four_eight_relationship() -> None:
    flat = _flat(_phase6_section())
    assert "32 rows" in flat
    assert "24 rows" in flat
    assert "8 of those" in flat


def test_documentation_describes_first_match_priority_classification() -> None:
    section = _flat(_phase6_section()).lower()
    assert "first matching rule" in section or "first-match" in section
    assert "higher-priority rule" in section


def test_documentation_describes_primary_categories_as_mutually_exclusive() -> None:
    section = _flat(_phase6_section()).lower()
    assert "mutually exclusive" in section
    assert "secondary symptoms may overlap primary categories" in section


def test_documentation_states_every_row_has_exactly_one_primary_category() -> None:
    section = _flat(_phase6_section()).lower()
    assert "contributes to exactly one primary failure category" in section
    assert "sum to 48" in section


def test_no_submission_rule_text_does_not_overclaim_the_full_symptom() -> None:
    """The rule must not read as if all 32 max_turn rows land in the 24-row category."""
    flat = _flat(_phase6_section())
    assert "did not submit, `termination_reason: `max_turns`, reward" not in flat
    assert "was not already classified by a higher-priority rule" in flat
