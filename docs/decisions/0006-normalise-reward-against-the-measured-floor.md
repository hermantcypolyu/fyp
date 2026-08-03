# 0006 — Normalise reward against the measured per-task floor

- **Status:** **Accepted** — implemented 2026-08-03
- **Date:** 2026-08-01 (accepted 2026-08-03)
- **Milestone:** raised in B2, implemented in B4
- **Affects:** `AGENT_BUILD_SPEC.md` §5.7 (`reward_v1`), §5.6 (one verifier for eval and
  reward), `benchmark/schema/task.schema.json`, `rewards/compose.py`

## Context

B2 finished with 24 tasks, all solvable by their gold patch and failable without it. But
`scripts/verify_gold_patches.py` also measures the **reward floor** — the fraction of hidden
tests that already pass against the untouched repository, i.e. what an agent scores for
submitting immediately without doing anything:

```
tasks              24
hidden tests       301
pass before start  139  (46.2%)
movable            162  (53.8%)
floor  min/mean/max 0.00 / 0.46 / 0.58
floor >= 0.50      11 tasks
floor >= 0.40      22 tasks
```

Source: `results/raw/b2_floors/report.json`, produced by
`uv run python scripts/verify_gold_patches.py --split all --json ...`.

Under §5.7, `reward_v1` is the raw pass fraction. So a do-nothing policy earns **0.46 on
average**, and 0.58 on the most generous task.

### Why the floor exists at all, and why it is not simply a bug

It is deliberate, and required. §5.4 rule 5 asks for real partial credit, and the authoring
guide asks every hidden suite to include tests that guard against a *plausible wrong fix* —
for example `bugfix_intword_power_rollover` keeps a test asserting that `999999999` still
renders as `"1.0 billion"`, because the tempting repair (delete the carry branch entirely)
would break it. Guard tests pass before the agent starts, by construction. A corpus with no
floor would be a corpus with no guards, and would reward exactly the lazy fixes we are trying
to detect.

So the floor is a design consequence, not an accident. The question is what the reward
function should do about it.

### What the floor actually costs us

It is worth being precise here, because the obvious argument is wrong.

GRPO computes its advantage *within a group of rollouts for the same prompt*:
$\hat{A}_i = (r_i - \mu_g)/\sigma_g$. A constant offset shared by every member of the group is
removed by $\mu_g$. **A per-task constant floor is therefore already invisible to GRPO's mean
subtraction.** The offset by itself does not bias the gradient.

The real costs are these:

1. **A zero-variance plateau, not an offset.** Doing nothing, editing the wrong file, making a
   syntactically valid but ineffective change, and running out of turns after reading files all
   score *exactly* the same number — the floor. These are not near each other; they are
   identical. When every rollout in a group lands on the plateau, $\sigma_g = 0$ and the group
   contributes no gradient at all. This is the same failure mode ADR 0005 rejected SWE-bench
   for, arriving from the opposite direction: there, everything scored 0; here, everything
   scores 0.46. A 1.5B model early in training will sit on this plateau often.

2. **Every reported number is uninterpretable.** "Mean reward 0.52" sounds like partial success
   and is in fact barely above free. Any figure in the dissertation computed from raw
   `reward_v1` — learning curves, pass@k on partial credit, before/after comparisons — carries a
   large, task-dependent constant that a reader cannot subtract without a table of floors.

3. **Tasks are not on a common scale.** Floors range from 0.00 to 0.58. Averaging raw reward
   across the corpus averages incommensurable quantities: gaining one test on
   `configfix_textkit_packaging` (9 movable tests) and on `bugfix_intword_power_rollover` (4
   movable) are worth 0.11 and 0.11 raw, but represent very different fractions of the available
   work. Cross-task aggregates and any per-task difficulty analysis inherit the distortion.

4. **It hard-codes an assumption about the algorithm.** Point 1 relies on GRPO's group
   normalisation. A value-function baseline (PPO), a global running-mean baseline, or any
   off-policy comparison across tasks does *not* remove a per-task constant. §5.7 is meant to be
   the reward for the project, not for one algorithm; the literature review (§6.2) explicitly
   compares GRPO against PPO and DPO, and a reward that only behaves under one of them
   prejudges that comparison.

## Options considered

1. **Do nothing.** Defensible for GRPO alone (see above), and it keeps §5.7 exactly as
   written. Rejected because of costs 2–4, and because the plateau in cost 1 is real regardless.
2. **Lower the floors by authoring.** Rewrite hidden suites so fewer tests pass untouched.
   Genuinely helps, and should be adopted as authoring guidance either way — but it cannot get
   the floor near zero without deleting the guard tests that make the corpus honest. Partial
   measure, not a solution.
3. **Normalise against the measured floor.** Proposed below.
4. **Binary reward** (1 if every hidden test passes, else 0). Kills the floor completely and is
   trivially interpretable. Rejected: it destroys partial credit, which §5.4 rule 5 requires and
   which is the only thing giving a 1.5B model a usable gradient on a task it cannot fully
   solve. This is ADR 0005's zero-variance problem again.
5. **Rank or leave-one-out baselines within the group.** Solves cost 1 elegantly but not costs
   2–4, and it makes the *reported* metric depend on the sampling procedure, which conflicts with
   §5.6's one-verifier invariant. Worth revisiting as an M7 ablation, not as the reward.

## Proposal

Store the measured floor in `task.yaml` and report reward relative to it.

### The formula, with one correction

The form suggested was:

```
reward = (passed - floor) / (n_tests - floor)
```

This is right in shape but mixes units: `passed` and `n_tests` are counts while `floor` as
currently reported is a fraction. Two equivalent, dimensionally consistent statements:

$$
r_\text{norm} \;=\; \frac{p - c}{n - c}
\qquad\text{(counts: } p \text{ passed, } n \text{ total, } c \text{ passing untouched)}
$$

$$
r_\text{norm} \;=\; \frac{f - \varphi}{1 - \varphi}
\qquad\text{(fractions: } f = p/n,\; \varphi = c/n)
$$

The count form is preferred as the stored quantity: $c$ is an exact integer that the validator
already measures, whereas $\varphi$ is a rounded float. Recommend a new field
**`verification.floor_tests`** (integer, $0 \le c < n$), sitting beside the existing measured
`verification.n_tests`.

`n - c > 0` is guaranteed: `check_behaviour` already rejects any task whose untouched repo
passes every hidden test (`NOT_FAILABLE`), so the denominator cannot be zero.

### Three consequences that must be decided together

These are the parts that are easy to get wrong, and the reason this is a proposal rather than a
patch.

**(a) Scores below the floor become negative.** An agent that deletes a function scores $p < c$,
so $r_\text{norm} < 0$. That is arguably correct — actively breaking the repository *should* rank
below doing nothing, and raw `reward_v1` currently cannot express that. Recommend **allowing
negative values, floored at $-1$**, rather than clipping at 0, precisely so "made it worse" and
"did nothing" stay distinguishable. This does mean `reward_v1` is no longer $\in [0,1]$ as §5.7
states, which is a spec amendment and needs your sign-off.

**(b) The hard-zero gates stop being a punishment.** This is the important one. Today a gate
yields `0.0`, which is far below every task's floor (0.38–0.58), so writing to `tests/**` is
strictly worse than doing nothing — as intended for what §5.7 calls "the #1 reward hack". After
normalisation, doing nothing *is* `0.0`. A gate that yields `0.0` would become exactly as good as
submitting immediately, and the deterrent evaporates. Recommend gates yield **$-1.0$**, i.e.
strictly below any achievable normalised score. Whatever we choose, the gate constant must be
defined relative to the normalised scale, not left at its old absolute value.

> **Coupling constraint — binding, not advisory.** Normalisation and the gate constant are **one
> change, and must land in a single commit.** Shipping the formula while leaving gates at `0.0`
> creates a window in which writing to `tests/**` and doing nothing score *identically* (both
> `0.0`), which silently removes the only deterrent against the #1 reward hack. The failure is
> invisible: nothing errors, no test goes red, and the gate-rate metric keeps logging hits while
> the reward stops caring about them. Any run performed inside that window is unusable and would
> have to be discarded, because its rewards cannot be reinterpreted after the fact.
>
> Two mechanical safeguards, to be added with the implementation:
> - `rewards/compose.py` derives the gate constant from the same module-level scale definition as
>   the formula, so the two cannot be changed independently.
> - A unit test asserts `gate_reward < normalised_reward(p=c)` — that is, a gated rollout is
>   strictly worse than a do-nothing rollout — which fails loudly if either lands alone.

**(c) Granularity gets worse on the tasks with the highest floors.** Normalisation rescales but
cannot create resolution. `bugfix_intword_power_rollover` has 4 movable tests out of 9, so its
normalised reward takes only 5 distinct values $\{0, 0.25, 0.5, 0.75, 1\}$; five tasks have 5
movable tests or fewer. This is an argument for adopting option 2 *as well* — treat "at least 6
movable tests" as an authoring target in `docs/task_authoring_guide.md`, alongside the existing
"floor below ~0.75" rule, and revisit the thinnest tasks.

### Keeping §5.6 intact

§5.6 requires that the same verifier produce evaluation metrics and the RL reward. Normalisation
must therefore happen in exactly one place — `rewards/compose.py` — and the trace schema should
record **both** numbers per rollout:

| field | meaning |
| --- | --- |
| `tests_passed`, `tests_total`, `floor_tests` | the raw measurement, never normalised |
| `reward_raw` | $p/n$, the §5.7 quantity, kept for continuity and for the dissertation's tables |
| `reward` | $(p-c)/(n-c)$, the number optimisation actually sees |

Reporting both costs nothing and makes every figure reconstructible either way.

### Implementation sketch (not done)

1. `task.schema.json`: add `verification.floor_tests`, integer, `minimum: 0`.
2. `validate_tasks.py`: `check_behaviour` already runs the untouched suite, so it can compare
   the declared `floor_tests` against the measurement and report a `FLOOR` error — and
   `--fix-counts` can write it back, exactly as it already does for `n_tests`. That makes the
   floor a *checked* measurement rather than a number an author types, and it means a hidden
   suite that drifts is caught rather than silently rescaling every past result.
3. `rewards/components.py` / `compose.py`: implement $(p-c)/(n-c)$, with the gate constant from
   (b), plus unit tests for $p = c$ (exactly 0), $p = n$ (exactly 1), $p < c$ (negative), and each
   gate.
4. `docs/reward_design.md`: record this ADR's reasoning as the v1 rationale.
5. Backfill `floor_tests` for all 24 existing tasks via `--fix-counts`.

## Decision

**Accepted 2026-08-03, all three points, implemented in a single commit.**

1. Reward range extends from $[0,1]$ to $[-1,1]$ — **yes**. `normalised_reward()` clamps at
   $-1$ rather than at $0$, so "made it worse" and "did nothing" stay distinguishable.
2. Hard-zero gates yield $-1.0$ rather than $0.0$ — **yes**, and shipped in the same commit as
   point 1, as the coupling constraint requires. `GATE_REWARD` is *derived* from `REWARD_MIN`
   in `rewards/compose.py` rather than written out again, so the two cannot drift apart, and
   `rewards/tests/test_compose.py::test_a_gated_rollout_scores_below_doing_nothing` asserts
   `GATE_REWARD < normalised_reward(p = c)` across five task shapes.
3. `task.yaml` gains `verification.floor_tests` — **yes**, backfilled as a *measured* value for
   all 36 tasks by `validate_tasks.py --fix-counts` and cross-checked against
   `verify_gold_patches.py`.

### As implemented

| where | what |
| --- | --- |
| `rewards/compose.py` | the scale, the formula, and `GATE_REWARD` derived from its bottom |
| `envs/coding_env/verifier.py` | calls `compose_reward()`; carries `floor_tests` on `VerifyResult` |
| `envs/coding_env/app.py` | `floor_tests` added to the `/verify` wire format |
| `analysis/trace_schema.json` | `schema_version` `1` → `2`; `reward` range `[-1,1]`; `floor_tests` recorded |

The denominator is `task.n_tests` — the count in git, not the count that ran — so shrinking it
still cannot pay; that hack is caught by `test_count_mismatch` and now costs $-1.0$.

Measured on the 36-task corpus after the change: do-nothing scores exactly $0.0$ and gold
exactly $1.0$ on **every** task, against a raw do-nothing mean of $0.467$ before. A gated
rollout scores $-1.0$ even when it solved the task outright.

One deviation from the sketch: `reward_raw` is not added as a new field, because
`result.pass_fraction` already *is* $p/n$ and was already recorded per rollout. Adding a second
name for the same number would have created two things to keep in sync for no gain.

## Consequences if accepted

- A do-nothing rollout scores exactly 0 on every task, so the plateau sits at a value that means
  what it says, and every reported mean becomes readable without a floor table.
- Tasks become comparable: 1.0 means "solved" and 0.0 means "achieved nothing" everywhere.
- The corpus gains a third measured invariant. `floor_tests` will drift whenever a hidden suite
  changes, and the validator will catch it — a small ongoing cost, and the same mechanism that
  already keeps `n_tests` and `repo.loc` honest.
- Historical raw numbers stay valid because `reward_raw` is retained; nothing already measured
  has to be re-run.
