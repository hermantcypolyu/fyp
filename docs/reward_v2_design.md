# `reward_v2` — design and component audit

- **Status:** A (correctness) and B (progress) implemented; C and D designed and **not shipped**
- **Date:** 2026-11-03
- **Supersedes nothing.** `reward_v1` (`rewards/compose.py`) is retained and is component A.
- **Code:** `rewards/v2.py`, `rewards/tests/test_v2.py`, `scripts/reward_distribution.py`

---

## 1. Why a second reward at all

`reward_v1` is a single number: the fraction of movable hidden tests won. ADR 0006 normalised
it against the measured floor, which fixed the *offset* problem — doing nothing scores 0.0
rather than 0.46 — but not the *plateau* problem. Doing nothing, editing the wrong file,
running out of turns, and crashing all still land on exactly 0.0.

For GRPO this is not a cosmetic complaint. The advantage is
$\hat{A}_i = (r_i - \mu_g)/\sigma_g$ over a group of rollouts sharing one prompt. When every
member lands on the plateau, $\sigma_g = 0$ and the group contributes no gradient at all.

This is measured, not theorised. `scripts/reward_distribution.py` over the 162 rollouts
currently on disk:

| | value |
|---|---|
| groups (run × task) with more than one rollout | 28 |
| of those, zero spread under `reward_v1` | **25** |
| — where every member is gated (reward cannot help) | 13 |
| — where progress could add up to 0.10 of spread | 12 |
| distinct runs | 53 |
| pooled mean / std | −0.481 / 0.500 |

Twenty-five of twenty-eight usable groups produce no gradient. Of those, thirteen are
groups where *every* member tripped a gate: every member scores `GATE_REWARD` whatever its
components say, so **no reward change can help those** — the fix is rollouts that differ.
The other twelve are genuinely reachable by component B. That distinction is the most useful
thing the offline measurement produced.

The pooled standard deviation of 0.500 is not reassurance: it is the spread *between* runs,
which GRPO never sees.

## 2. The components

| | component | weight | status | measures |
|---|---|---|---|---|
| **A** | correctness | 0.60 | shipped | hidden-test fraction won, normalised to the floor |
| **B** | progress | 0.20 | shipped | best-seen improvement in passing *visible* tests |
| **C** | validity | — | deferred | did the run respect the rules (largely already gates) |
| **D** | tool | — | deferred | tool-call correctness and error rate |
| **E** | efficiency | — | **rejected for training** | turns, tokens, runtime |

### A — Correctness, 0.60

Unchanged from `reward_v1`: `(passed − floor) / (n_tests − floor)`, clamped to
`[−1, 1]`. Computed only by the private verifier, over tests the agent never sees.

It carries the largest weight because it is the only component that measures whether the
*problem was solved*. Everything else measures how the agent behaved on the way.

### B — Progress, 0.20

The best-seen improvement in passing tests across the episode's `run_tests` calls.

Three decisions are load-bearing:

1. **It reads the smoke suite only.** That suite is already visible to the agent, so scoring
   it leaks nothing — the agent could print the same number itself. The hidden suite is
   never consulted.
2. **Best-seen, not final-state.** An agent that improved and then reverted keeps credit for
   the improvement. The revert is already priced by component A; charging it twice would
   penalise exactly the exploration the reward is meant to encourage.
3. **Measured against the first observation, not zero.** The agent did not author the repo,
   so tests passing before it touched anything are not progress it earned.

**Why 0.20 and not more.** A rollout that made perfect progress but did not solve the task
scores at most 0.20; a rollout that solved it with no progress scores 0.60. The ordering
cannot invert. This bounds the cheapest hack — "learn to run the smoke tests" — below the
reward for actually fixing the bug.

**Gated behind non-negative correctness.** A plain weighted sum would let a regressing
rollout buy itself above zero: correctness −0.25 × 0.60 = −0.15, and perfect progress adds
+0.20, landing at +0.05. The agent would be *paid* for activity that made the code worse, and
would outscore a rollout that did nothing. Progress is therefore only credited at or above
the do-nothing line. Tripwire:
`test_a_negative_correctness_with_perfect_progress_is_still_negative`.

### C — Validity *(deferred)*

Whether the submission was well-formed: submitted at all, patch applies, syntax parses,
non-empty change, no out-of-scope writes.

**Almost all of it is already a gate.** `no_submit`, `syntax_broken`, `tests_modified`,
`sandbox_escape`, `invalid_patch` and `forbidden_path_modified` all exist and all hard-zero
the run. A gate is a stronger statement than a soft penalty: it cannot be traded against
correctness. Shipping C as a weight on top of the gates would score the same failures twice.

What remains genuinely soft — a patch that applies but is an empty no-op — is rare enough to
measure before it is priced. Deferred to Phase 4.

### D — Tool behaviour *(deferred)*

Tool-call format correctness, tool error rate, invalid arguments, repeated no-op operations.

Deferred with C, and for the same structural reason: shipping two new weighted components at
once, while the 1.5B model's variance is still zero, would move two dials that cannot be
attributed separately. It also carries the highest risk of encoding a preferred *strategy*
rather than an outcome, so it should be designed after the outcome components are measured.

### E — Efficiency *(rejected as a training reward)*

Turns, tool calls, tokens, runtime, repeated reads.

**Rejected, not deferred.** An efficiency term in the reward teaches the agent to submit
sooner, and the fastest way to submit sooner is to submit without working. Since the task
budget is already enforced server-side (`max_tool_calls`) and agent-side (`max_turns`), an
efficiency term adds no information the harness does not already enforce — it only adds an
incentive to stop early.

Efficiency remains a useful **evaluation metric**: turns-to-solve, tokens-per-solved-task.
That is the training/evaluation split in §4.

---

## 3. What happens to a gated run

A gate returns `GATE_REWARD` (−1.0, `REWARD_MIN`) as `final`, no matter what the rollout
passed or how far it progressed. Unchanged from v1, and for the same reason: if a gate could
be bought back with progress, it would stop being a deterrent.

The component fields are still filled in. That is deliberate — a post-mortem needs to see
that a gated run had in fact solved the task before the gate discarded it, which is the
difference between "the model is bad" and "a guard fired".

## 4. Training reward vs evaluation metric

This split must stay explicit, or the report will describe a metric the training loop never
optimised, or optimise something the report never measures.

| quantity | training reward | evaluation metric |
|---|---|---|
| hidden-test pass fraction | ✅ A | ✅ |
| best-seen progress | ✅ B | ✅ |
| pass@1 | ❌ | ✅ primary |
| gates fired / gate rate | ✅ (hard-zero) | ✅ |
| turns to solve | ❌ (rejected) | ✅ |
| tokens per solved task | ❌ (rejected) | ✅ |
| tool error rate | ❌ (deferred) | ✅ |
| reward variance within a group | ❌ | ✅ diagnostic |

Pass@1 is the headline result and is deliberately **not** in the training reward: it is
binary, and a binary reward is ADR 0005's zero-variance failure again.

## 5. Duplicate-metric audit

For each component: what it measures, what raw data it needs, what it overlaps, and whether
the agent can manipulate it.

### A — Correctness

| question | answer |
|---|---|
| What phenomenon? | Was the problem solved |
| Raw data needed | hidden-test per-test results + `task.yaml` counts. **Private.** |
| Overlaps another component? | No — B is visible-test progress, A is hidden-test outcome. A rollout can gain B and lose A. |
| Can the agent manipulate it? | Only by (a) passing hidden tests, which is the goal, or (b) shrinking the denominator, which `test_count_mismatch` gates. |
| Training or evaluation? | Both |
| Behaviour change from reweighting up | More focus on the final patch, less on exploration |

### B — Progress

| question | answer |
|---|---|
| What phenomenon? | Did the agent get closer to a solution before stopping |
| Raw data needed | per-`run_tests`-call passing counts. **Visible smoke suite only.** |
| Overlaps another component? | Partially with A: a rollout that gains progress usually gains some correctness. The overlap is bounded because B reads a *different suite* — an agent can pass every smoke test and still fail every hidden test, which is a common 1.5B failure mode (`bugfix_intcomma_zero_ndigits`: 6/11 smoke, floor 6/11 hidden). |
| Can the agent manipulate it? | It can run `run_tests` more often — but best-seen, not cumulative, so repetition earns nothing. It cannot edit the smoke suite (`forbidden_write_globs`), and the baseline is the first observation. |
| Training or evaluation? | Both |
| Behaviour change from reweighting up | More iteration and self-checking; if raised past A, would teach test-passing without fixing |

**The duplication risk, stated honestly.** If progress and correctness move together for
almost every rollout, B adds variance that A already provided, and the weight is wasted
rather than harmful. This can only be settled by measuring both on rollouts produced *after*
the toolbox starts recording counts. It cannot be settled from the data on disk.

---

## 6. What the offline measurement does and does not show

`scripts/reward_distribution.py` scores all 162 historical rollouts under both rewards.

**What it shows (measured):**

- 25 of 28 multi-rollout groups have zero spread under `reward_v1`.
- 13 of those 25 are all-gated groups, where every member scores `GATE_REWARD` by
  construction. **The reward cannot rescue them**; only rollouts that differ can.
- The remaining 12 are non-gated and reachable by component B.
- The pooled v1 statistics: 162 rows, mean −0.481, std 0.500.

**Progress is recoverable after all — and reads zero.** An earlier revision of this section
claimed that "no rollout on disk recorded per-turn test counts", so that B was `None`
unrecoverable. That claim was wrong. Every `run_tests` call already puts its summary in
`ToolResult.output` as `"N/M passed (smoke tests only; ...)"`, and the rollout JSONL keeps
the full tool transcript. Re-scraping all 162 rows (`smoke_delta` over each row's `output`
list) gives:

| Measurement | Count |
|---|---|
| rollout rows on disk | 162 |
| rows with at least one readable `N/M passed` | **95** |
| rows where `best − first > 0` | **0** |
| distinct count-sequences observed across all rows | 0 with more than one reading |
| rollouts that call `run_tests` more than once | **4** |
| of those, an edit between the first and last test run | 2 |

So B is not blocked by an instrumentation gap. It is blocked by **behaviour**: across 95
rollouts that ran the smoke suite, not one ever ended with more passing tests than it started
with. 74 of them report `2/2 passed` on their single test call — the suite already passed
before the agent touched anything, and the agent then ran it once and stopped. The reward
cannot reward progress that never happens.

**What this means for B.** The component is correct and tested, but its expected value on
current model behaviour is zero. Shipping A+B as specified therefore changes almost nothing
on this data: for the 95 rows with counts, `progress = 0.0`, and `0.0 × 0.20` contributes
nothing. This is a real result, not a null one — it narrows the problem from "the reward is
uninformative" to "the model does not perform the edit-then-retest loop at all", which is
the thing the training run has to teach. It also raises the value of the tool-behaviour
component D, deferred here, which would reward *attempting* the loop.

**What it does not show, and must not be claimed to show:**

- **That the weights are right.** 0.60/0.20 is a defensible starting point, not a tuned one.
- **That B will fire in a live run.** Nothing on disk shows it firing even once. The claim
  that progress *can* create variance remains a design argument backed by a unit test
  (`test_progress_creates_spread_in_a_group_that_was_degenerate`), not a measured result.

**The ceiling on what progress can contribute.** With `final = 0.60·A + 0.20·B`, a group
where every member has the same `A` gains spread only from `B`:

$$\sigma_{final} = 0.20\,\sigma_B$$

`B` is capped at 1.0 and floored at 0.0, so the largest possible $\sigma_B$ occurs when half
the group has `B = 0` and half has `B = 1`, giving $\sigma_B = 0.5$ and

$$\sigma_{final} \le 0.20 \times 0.5 = 0.10$$

Verified numerically in `scripts/reward_distribution.py`'s `max_progress_std` column: 0.10
for each of the 12 reachable groups, 0.0 for each of the 13 all-gated ones.

So progress can give a zero-variance group a gradient, but a modest one — at most 0.10 of
spread where it had 0.00, well below the 0.50 the *pooled* data shows. That is a real gradient
where there was none; it is not a large one, and the dissertation should not describe it as
one.

## 7. Open items carried forward

1. ~~Record per-turn test counts in `Toolbox.log`.~~ **Closed — the data already exists.**
   `run_tests` writes `"N/M passed"` into `ToolResult.output`, and `smoke_delta` reads it
   from the rollout JSONL with no environment change. What was missing was not the
   instrumentation but the extractor, which now exists (`rewards/v2.py::smoke_delta`) and
   is wired into `scripts/reward_distribution.py`.
2. **Measure the A/B overlap** on rollouts produced after a training run. If the correlation
   is near 1, reconsider the 0.20 weight.
3. **Design C and D** once A+B is shown measurable. The measurement above makes D (tool
   behaviour) more urgent than it looked: B reads zero because the edit-then-retest loop is
   essentially never attempted (4 rollouts in 162), and D is the component that could price
   attempting it.
4. **All-gated groups stay degenerate** (see §6). That is a rollout-diversity problem, not a
   reward problem.
5. **The zero-progress finding needs to reach the training run, not just this document.**
   Any future plan that assumes B will separate groups on current behaviour is contradicted
   by 95 rows of evidence.
