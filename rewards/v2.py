"""`reward_v2`: correctness plus observed progress, shipped two components at a time.

`reward_v1` (`rewards/compose.py`) is not replaced. It is component **A** of this module and
is still the only thing that decides whether a rollout solved the task. What v2 adds is a
second, separately reported number for the behaviour v1 cannot see: *whether the agent was
getting anywhere before it ran out of road.*

**Why progress is worth 0.20 and not more.** A rollout that ends at the floor has an
advantage of exactly zero under GRPO's group normalisation (ADR 0006), so a 1.5B model
spends most of its early training contributing nothing. Progress gives it a gradient on
rollouts that improved without solving. But progress is *not* success: an agent can raise
the smoke-test count and never submit a working patch. If progress were weighted anywhere
near correctness, the cheapest way to raise reward would become "learn to read tests", not
"learn to fix code". At 0.20, correctness alone (1.0) outranks a perfect progress score
(1.0 * 0.20), and can never be offset by it.

**Why A and B ship first, and C and D do not.** C (submission validity) and D (tool
behaviour) are both largely expressible as gates, which already exist and already zero the
run. Shipping them as *weights* before the reward distribution has been measured would move
two dials at once while the 1.5B model's variance is still zero, and a resulting change in
behaviour could not be attributed to either. They stay `None` in `reward_components` (not
`0.0` - see the schema's note) until Phase 4 shows A+B is measurable at all.

**What "progress" reads, and why the data does not yet exist.** Progress is the best
per-turn improvement in *visible* tests-passed across the episode's `run_tests` calls. Two
properties make it safe to score:

- It reads the **smoke suite only**. That suite is already visible to the agent, so scoring
  it leaks nothing: the agent could have printed the same number itself. The hidden suite
  is never consulted, and `smoke_delta` has no way to reach it.
- It is a **best-seen** delta, not a final-state delta, so undoing a good intermediate edit
  is not punished twice - the patch already carries that cost in component A.

The number does not exist in any rollout on disk yet. `Toolbox.log` records `ToolCall`
objects with `ok` and `output` but no test counts, and the trace `steps[]` added in Phase 2
carries a `tool_result_summary` string rather than a count. So `progress_reward` here is the
scoring rule, and `smoke_delta` is the extractor that will feed it once the toolbox records
counts. Wiring the toolbox is deliberately a separate change - it touches the environment,
and this module is pure so that the arithmetic can be tested without one.

**Where progress is allowed to count.** Only for a rollout whose correctness is at or above
the do-nothing line. A plain weighted sum would let a regressing rollout buy itself above
zero - it would be paid for activity that made the code worse, and would outscore a rollout
that did nothing. `_fuse` gates on that, and
`rewards/tests/test_v2.py::test_a_negative_correctness_with_perfect_progress_is_still_negative`
is the tripwire.

**The one number that must not drift.** `compose` returns the pair and `final` is computed
from the weights *in this module*. `analysis/trace.py` asserts the fusion it writes into
`reward_components.final` equals `result.reward`; if these weights were restated anywhere
else, that assert would fire during a run rather than in a unit test.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from rewards.compose import GATE_REWARD, REWARD_DID_NOTHING, REWARD_MIN, REWARD_SOLVED, normalised_reward

__all__ = [
    "CORRECTNESS_WEIGHT",
    "MAX_SMOKE_DELTA",
    "PROGRESS_WEIGHT",
    "RewardV2",
    "SHIPPED_COMPONENTS",
    "WEIGHTS",
    "compose_v2",
    "progress_reward",
    "smoke_delta",
]

#: Correctness dominates, and the weights sum to 1.0 so `final` stays on the scale v1 defined.
#: A rollout that solved the task perfectly and made no measured progress scores
#: CORRECTNESS_WEIGHT (0.60) once progress is measured; a rollout that made perfect progress
#: and did not solve it scores at most PROGRESS_WEIGHT (0.20).
CORRECTNESS_WEIGHT = 0.60
PROGRESS_WEIGHT = 0.20

#: Components C and D are designed but not shipped. They are absent from `WEIGHTS` rather
#: than weighted at 0.0, so that "not yet measured" cannot be mistaken for "measured as zero".
SHIPPED_COMPONENTS = ("correctness", "progress")

WEIGHTS: dict[str, float] = {
    "correctness": CORRECTNESS_WEIGHT,
    "progress": PROGRESS_WEIGHT,
}

#: The largest number of newly-passing smoke tests that can count as progress within one
#: episode. Not a property of any task: it bounds the component so that a rollout which
#: happens to sit in a repo with hundreds of visible tests cannot dwarf one with eight.
MAX_SMOKE_DELTA = 10


@dataclass(frozen=True)
class RewardV2:
    """The v2 decomposition. `final` is what the training loop reads; the rest is diagnosis."""

    correctness: float
    progress: float | None
    final: float
    gates: tuple[str, ...] = ()

    def to_components(self) -> dict[str, float | None]:
        """The `reward_components` object of `analysis/trace_schema.json`, exactly.

        C and D are `None` because they are not shipped. Writing `0.0` here would assert
        that they were measured and scored nothing, which is a different claim.
        """
        return {
            "correctness": self.correctness,
            "progress": self.progress,
            "validity": None,
            "tool": None,
            "final": self.final,
        }


def progress_reward(best_delta: int | None) -> float | None:
    """Map a best-seen improvement in passing visible tests onto `[0, 1]`.

    `None` in, `None` out: an episode whose smoke suite never ran has no measurable
    progress, which is not the same as measuring zero progress. Returning 0.0 there would
    silently punish a task that ships no smoke tests at all.
    """
    if best_delta is None:
        return None
    capped = min(max(best_delta, 0), MAX_SMOKE_DELTA)
    return capped / MAX_SMOKE_DELTA


def compose_v2(
    passed: int,
    n_tests: int,
    floor_tests: int = 0,
    *,
    gates: Iterable[str] = (),
    progress: float | None = None,
) -> RewardV2:
    """Score one rollout under `reward_v2`.

    **Gates still win outright.** A gated run returns `GATE_REWARD` as its final score no
    matter what it passed or how far it progressed - the same hard-zero as v1, and for the
    same reason: if a gate could be bought back with progress, the gate would stop being a
    deterrent. The component fields are still filled in, so a post-mortem can see that a
    gated run did in fact improve before it was discarded.

    `progress` is the already-scaled value from `progress_reward`, or `None` when the run has
    no measurable progress. `None` contributes nothing to the sum rather than being
    redistributed onto correctness - see `_fuse` for why a task-dependent scale would be
    worse than an unreachable top.
    """
    ordered_gates = tuple(dict.fromkeys(gates))
    correctness = normalised_reward(passed, n_tests, floor_tests)

    if ordered_gates:
        return RewardV2(
            correctness=correctness,
            progress=progress,
            final=GATE_REWARD,
            gates=ordered_gates,
        )

    final = _fuse(correctness, progress)
    return RewardV2(correctness=correctness, progress=progress, final=final, gates=ordered_gates)


def _fuse(correctness: float, progress: float | None) -> float:
    """Weighted sum of the shipped components, clamped to the scale v1 defined.

    **Progress is gated behind non-negative correctness.** A plain weighted sum lets a
    regressing rollout buy itself above zero: correctness -0.25 weighted at 0.60 is -0.15,
    and a perfect progress term adds +0.20, landing at +0.05. A rollout that broke more tests
    than it fixed would then be rewarded *more* than one that did nothing - the reward would
    be paying for activity that made the code worse. So the progress term is only credited
    when correctness is at or above the do-nothing line; below it, the score is the
    correctness component alone.

    When `progress` is `None` the progress weight is **not** redistributed onto correctness.
    Redistributing would mean a task without a smoke suite scores a solved rollout at 1.0
    while a task with one scores it at 0.60 for the identical patch - the reward would
    depend on the corpus rather than on the agent. `None` therefore contributes 0.0, which
    is an honest statement: correctness is the only thing measured there.

    The clamp matters in one direction: 0.60 * -1.0 is -0.60, and nothing above can push the
    sum past `REWARD_SOLVED`. The clamp is applied anyway so a future component cannot push
    `final` off the scale that `GATE_REWARD` and the schema's bounds assume.
    """
    if progress is None or correctness < REWARD_DID_NOTHING:
        value = CORRECTNESS_WEIGHT * correctness
    else:
        value = CORRECTNESS_WEIGHT * correctness + PROGRESS_WEIGHT * progress
    return max(REWARD_MIN, min(REWARD_SOLVED, value))


_PASSED_OF = re.compile(r"(\d+)\s*/\s*(\d+)\s+passed")


def smoke_delta(summaries: Iterable[str]) -> int | None:
    """Best-seen improvement in passing smoke tests over a sequence of `run_tests` summaries.

    The summaries are the first line of `ToolResult.output` for a `run_tests` call - e.g.
    `"6/11 passed (smoke tests only; these are not the grading tests)"`. The baseline is the
    **first** readable summary, not zero: the agent did not author the repo, so tests that
    passed before it touched anything are not progress it earned. A rollout that never ran
    the smoke suite yields `None`.

    Returns the largest `passed - baseline` seen, floored at 0. Undoing a good edit and
    landing lower than the baseline is not scored as negative progress: the damage is
    already in the final patch and is priced by component A.
    """
    counts: list[int] = []
    for line in summaries:
        match = _PASSED_OF.search(line or "")
        if match:
            counts.append(int(match.group(1)))
    if not counts:
        return None
    return max(0, max(counts) - counts[0])
