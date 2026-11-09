# 0008 — Phase 7 uses the pinned 3B arm rather than the 1.5B arm

- **Status:** **Accepted**
- **Date:** 2026-11-09
- **Milestone:** Phase 7 (one-step GRPO + LoRA feasibility smoke)
- **Affects:** `configs/training/*`, `docs/CURRENT_PROJECT_STATUS.md`,
  `docs/PHASE6M_TEST_RUNNER_CORRECTION.md`
- **Supersedes nothing.** Extends the Phase 6 closure recorded in
  `docs/PHASE6L_MODEL_CAPACITY_3B.md` and `docs/PHASE6M_TEST_RUNNER_CORRECTION.md`.

## Context

Phase 6 collected two infrastructure-valid four-rollout groups from the same task,
`bugfix_offbyone_pagination`, under four distinct sampling seeds. The arms differ only in base
model. Every value below is produced by the repository's own implementation
(`training/grpo_math.py::group_advantages`), not by hand arithmetic.

### The 1.5B group (Phase 6K)

Rewards: `[-1.0, -1.0, -1.0, -1.0]`.

`group_advantages()` raises `DegenerateGroupError` on this group. That is deliberate: the module
rejects the group rather than returning a list of zeros. `scripts/run_optimizer_smoke.py`
converts the exception into a `RuntimeError`, so the run **fails fast**. It does not skip the
degenerate group and continue, and it does not record a skipped-group status.

The reason is structural, not incidental. GRPO computes `(reward - group_mean) / (group_std +
epsilon)`. When every member carries the same reward, the numerator is zero for all four
members. There is no ordering to learn from, so there is no advantage and no gradient. The
implementation refuses rather than emitting a loss with no gradient that would look like
training.

**Consequence: no optimizer update is permitted from the 1.5B group.**

### The 3B group (Phase 6M, after the test-runner correction)

Rewards: `[-1.0, -1.0, 0.0, -1.0]`.

| Quantity | Value |
| --- | --- |
| Group mean reward | `-0.75` |
| Population reward standard deviation | `0.4330127018922193` (≈ `0.4330`) |
| Normalised advantages | `[-0.5773, -0.5773, +1.7320, -0.5773]` |
| Advantage mean | ≈ `0` (`-5.55e-17`) |
| Advantage population standard deviation | `0.9999976906042567` (≈ `1`) |
| Reward-eligible rows | `4/4` |
| Infrastructure-invalid rows | `0` |
| Successful `run_tests` executions | `2` |
| Non-empty submissions | `2` |
| Solved rollouts | `0/4` |

The normalisation divides by the **population** standard deviation (the group *is* the
population; it is not a sample estimate of a wider spread) and adds `ADVANTAGE_EPSILON = 1e-6`
to the denominator before dividing. That epsilon is why the advantage standard deviation is
`0.99999769...` rather than exactly `1.0`; the inflation is the price of the guard that turns a
near-zero spread into a `DegenerateGroupError` instead of a division by ~zero.

The `0.0` rollout carries the maximum advantage `+1.7320`. The three floor rollouts each carry
`-0.5773`. Advantage ordering matches reward ordering.

## Decision

1. Use the already validated, pinned `Qwen/Qwen2.5-Coder-3B-Instruct` arm
   (revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`) for **exactly one** Phase 7
   optimizer-step GRPO + LoRA feasibility smoke.
2. Do **not** use the 1.5B arm for Phase 7 training. It cannot produce an update.
3. Permit **exactly one** optimizer step initially.
4. Require a **separate review** before any additional iteration.
5. **Never** download, load, evaluate, fine-tune or train a 7B model. This prohibition is
   permanent.

## Interpretation

- This is **not** a continuation of model-capacity comparison. Model-capacity comparison is
  **closed**; it was performed once (1.5B → 3B) and recorded in
  `docs/PHASE6L_MODEL_CAPACITY_3B.md`. Phase 7 does not reopen it and does not extend it.
- The 3B arm is selected because it is the **only tested arm that produced a non-degenerate
  reward group accepted by the repository's GRPO mathematics**. It was not selected for being
  better at the task — it is not demonstrably better at the task.
- The experiment tests **training machinery**, not task-solving ability.
- The learning signal is **thin**: exactly one rollout sits above the reward floor, so the
  within-group contrast is carried by a single member. A group of this shape is adequate to
  prove that a finite non-zero update can complete. It is a weak basis for establishing that
  anything useful is learned.
- `0/4` rollouts solved the task. The `0.0` rollout is a partial score on a semantically
  incorrect patch, not a fix.

## Consequences

- One finite, non-zero LoRA update may be tested. Success means the machinery works, not that
  the model improved.
- Base-model parameters must remain frozen. A changed base tensor is an integrity failure, not a
  training result.
- Only intended LoRA parameters may change.
- Checkpoint save and reload must be verified, and the checkpoint remains diagnostic. It is not
  promoted to a formal trained checkpoint.
- No multi-iteration training is authorised by this ADR. Additional iterations require a
  separate review and a separate decision record.
- No claim of model-performance improvement may be made from this experiment.

## Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| 1.5B Phase 7 arm | `group_advantages()` raises `DegenerateGroupError` on the uniform floor group, so no optimizer update is possible. |
| Immediate 2-to-5-iteration training | Premature. Nothing is yet known about whether the machinery completes a single update, and a multi-iteration run would confound machinery failure with signal scarcity. |
| Any 7B model | Permanently prohibited. |

## References

- `docs/PHASE6L_MODEL_CAPACITY_3B.md` — the closed capacity comparison.
- `docs/PHASE6M_TEST_RUNNER_CORRECTION.md` — the test-runner fix and the rerun that produced the
  accepted 3B group.
- `training/grpo_math.py` — `group_advantages`, `DegenerateGroupError`, `ADVANTAGE_EPSILON`.
- `docs/decisions/0006-normalise-reward-against-the-measured-floor.md` — the reward floor that
  makes `-1.0` a meaningful value rather than a failure sentinel.
- `docs/decisions/0007-multi-turn-rollout-adapter-and-unscorable-rollouts.md` — why an
  unscorable rollout must not be silently scored `0.0`.
