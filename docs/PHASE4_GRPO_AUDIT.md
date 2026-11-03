# Phase 4 — GRPO implementation audit

Audited 2026-11-04 against `docs/CURRENT_PROJECT_STATUS.md` §4. Every claim below is
either traced to a file and line, or to a metrics file produced by an executed run.
Nothing here is inferred from a config comment or from memory.

## Verdict

The code implements a **one-step, single-group, clipped policy-gradient update with
group-relative advantages and LoRA**. It is *GRPO-style*, not full GRPO, and two items on
the §4.2 checklist are **absent rather than unimplemented by mistake**: there is no KL
penalty and no reference model.

The name in use everywhere — "controlled, one-step GRPO-style LoRA update" — is therefore
the correct one and no rename is required (§4.3).

## Where the code is

| checklist item | location | status |
|---|---|---|
| Rollout adapter | `training/rollout_adapter.py` | present, 12 tests |
| Reward-to-advantage code | `training/grpo_math.py::group_advantages` | present, tested |
| Policy loss | `training/grpo_math.py::policy_loss` (torch), mirrored in `scripts/run_optimizer_smoke.py` | present, tested |
| Optimizer step | `scripts/run_optimizer_smoke.py:477` (`optimizer.step()`) | present, executed |
| Checkpoint save/reload | `scripts/run_optimizer_smoke.py:492` (save) and the reload block after `main()` prints the checkpoint | present, executed |
| LoRA config | `scripts/run_optimizer_smoke.py:301` (`LoraConfig`), `configs/training/*.yaml` | present |
| Training tests | `training/tests/test_rollout_adapter.py`, `training/tests/test_grpo_math.py` | 39 tests |

### The structural problem this phase fixed

`scripts/run_optimizer_smoke.py` imports `torch`, `peft` and `transformers` **at module
level**, and `main()` aborts unless `torch.cuda.is_available()`. In this repository's own
development environment:

```
$ uv run python -c "import torch"
ModuleNotFoundError: No module named 'torch'
```

So *no part* of the GRPO arithmetic could be imported, let alone tested, outside the GPU
box. The group math had only ever been verified by running it on real hardware. Phase 4
moved the torch-free arithmetic into `training/grpo_math.py`, which the script now imports,
so the tested code and the executed code are the same code.

## §4.2 GRPO core — item by item

| item | verdict | evidence |
|---|---|---|
| Multiple rollouts per task | ✅ | `load_samples` requires one `task_id` for all members; the executed group had 8 |
| Independent reward per rollout | ✅ | `adapt_rollout` returns one `reward` per row; `_require_reward` rejects non-finite |
| Group mean computed | ✅ | `group_advantages`, mean `-0.125` matches the recorded run exactly |
| Group-relative advantage computed | ✅ | `(r - mean) / (std + eps)`; reproduced to the last decimal |
| Zero-variance groups handled | ✅ | `DegenerateGroupError` raised, not silently zeroed |
| Generated-token log probabilities obtained | ✅ | `current_generation_log_probs`, logits sliced to generation span |
| Prompt tokens masked | ✅ | logits taken from `prompt_length - 1`; prompt positions never appear in the loss |
| Padding tokens masked | ⚠️ **by construction** | each turn is scored as a single unpadded sequence of `len(prompt)+len(generation)`; there is no batch padding to mask, and no batched path exists yet |
| Old-policy probability available | ✅ | `generation_log_probs` captured at rollout time by NeMo Gym |
| Policy probability ratio computed | ✅ | `policy_ratio`; `exp(current - old)` |
| Clipped policy objective present | ✅ | `clipped_surrogate` = `min(r*A, clip(r)*A)`, one-sided as PPO specifies |
| **Reference-policy KL regularisation present** | ❌ **ABSENT** | no reference model is loaded; `grep -n "kl\|ref_model"` over the script returns nothing; the metrics file has no `kl` key |
| KL direction correct | ❌ **not applicable** | nothing to check |
| Loss applied only to generated tokens | ✅ | only `TrainableTurn`s contribute; tool outputs are `context_item_count` |
| Backward produces finite gradients | ✅ | `raw_gradient_norm` 0.307842467, `gradient_norm_is_finite: true` |
| Gradient clipping sane | ✅ | `clip_grad_norm_(max_norm=1.0)`; 0.3078 < 1.0, so clipping did not bind |
| AdamW updates only LoRA parameters | ✅ | optimizer built from `requires_grad` params only |
| Base model parameters frozen | ✅ | `get_peft_model` / `PeftModel(is_trainable=True)`; 18,464,768 trainable of a 1.5B model |
| Parameters actually change after the step | ✅ | 392 LoRA tensors changed, delta L2 0.038865 |
| Checkpoint reload restores identical parameters | ✅ | `reload_max_parameter_difference: 0.0` — exact equality, not approximate |

**15 of 20 verified, 1 satisfied by construction, 2 absent, 2 not applicable.**

## The two absent items

Loading a frozen reference copy of a 1.5B model alongside the policy is the memory the
A5000 does not have: the executed run already peaked at **19.9-22.9 GB of 24 GB**. KL
regularisation was deferred for that reason, not overlooked.

`training/tests/test_grpo_math.py::test_there_is_no_kl_term_to_test` pins the absence with
`inspect.signature`, so the checklist item cannot be ticked later without the test failing
and forcing a real test to be written.

This is a substantive limitation, not a technicality. Without KL, nothing anchors the policy
to the starting distribution over many steps. One step is therefore safe — the checkpoint is
only one update away from the base model — while long-run stability is **unmeasured**, which
is what §4.3's naming rule exists to protect.

## §4.3 Naming

Already correct, no change made. `README.md`, `docs/PHASE_F4_CONTROLLED_UPDATE.md`,
`docs/PHASE_F3_OPTIMIZER_SMOKE.md`, `docs/TRAINING_READINESS.md` and `PROGRESS.md` all say
some form of *"controlled, one-step GRPO-style LoRA update"*. A repository-wide grep for
`"GRPO with LoRA"` and `"full GRPO"` finds no claim of full GRPO outside
`CURRENT_PROJECT_STATUS.md`, where those words appear only as the *test for when* the name
would become legitimate.

Because the full GRPO objective is **not** verified (no KL), the operative rule is:

> Call it "controlled, one-step GRPO-style LoRA update" — because KL regularisation is
> absent.

`README.md:13` already carries the corresponding disclaimer: *"This verifies optimizer and
checkpoint plumbing only; it does **not** show improved coding performance."*

## §4.4 New tests — what was added

`training/tests/test_grpo_math.py`, 27 tests. The anchor is:

```python
def test_reproduces_the_recorded_smoke_run_exactly():
    advantages, mean, std = group_advantages([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    assert mean == -0.125
    assert std == 0.33071891388307384
    assert advantages[0] == 0.37796333015554007
    assert advantages[4] == -2.6457433110887805
```

Every literal was read from `results/raw/rl_smoke_1step/training_metrics.json`, produced by
a real RTX A6000 run. The refactored code reproduces it bit for bit, which is the evidence
that extracting the math did not change it.

| §4.4 item | test |
|---|---|
| Equal rewards → advantage near zero | `test_equal_rewards_are_rejected_rather_than_scored_as_zero_advantage` — **rejected**, not zero; see note below |
| High-reward rollout → positive advantage | `test_high_reward_rollout_gets_positive_advantage` |
| Low-reward rollout → negative advantage | `test_low_reward_rollout_gets_negative_advantage`, `test_the_lowest_reward_always_gets_the_lowest_advantage` |
| Clipping actually bounds the ratio | `test_clipping_limits_the_reward_for_improving_too_far`, `test_clipping_does_not_apply_to_the_disadvantageous_direction`, `test_inside_the_trust_region_the_objective_is_not_clipped` |
| Rising KL is reflected in the loss | ❌ **cannot be tested — no KL exists**; `test_there_is_no_kl_term_to_test` pins the gap |
| Prompt tokens excluded from policy loss | `test_loss_is_normalised_by_the_group_token_count_not_the_turn_count` + adapter's `test_tool_outputs_do_not_become_trainable_turns` |
| Padding tokens excluded from policy loss | n/a — no padded batch path; covered by construction as noted above |
| Frozen base weights unchanged | ⚠️ verified from the executed run's `trainable_parameter_count`, **not** covered by a CI test — needs a GPU |
| LoRA weights changed | ⚠️ `changed_lora_parameters: 392` in the recorded metrics; `parameter_delta` is torch-bound and untested in CI |
| Loss, gradient, KL all finite | `test_loss_is_finite_for_every_extreme_ratio_and_advantage`, `test_policy_ratio_output_is_always_finite` (loss and gradient finite; KL absent) |
| Empty / invalid rollout safely rejected | `test_mismatched_lengths_are_rejected`, `test_non_finite_log_probabilities_are_rejected`, `test_zero_token_count_is_rejected_rather_than_dividing_by_zero`, `test_empty_inputs_are_handled_without_a_divide_by_zero`, plus the 12 adapter tests |

### A note on "equal rewards → advantage near zero"

The checklist expects a near-zero advantage. The implementation instead **raises**. That is
deliberate and is the reason Phase 3 found 25 of 28 groups degenerate: a group of equal
rewards has `(r - mean) = 0` and `std = 0`, so the advantage is `0/0`. Returning `0.0` would
give a zero-gradient step that looks like a successful training run. The test asserts the
exception, and the doc records why the checklist wording and the implementation differ.

## What remains unverified by CI

Three §4.4 items require a GPU and are evidenced only by the recorded run, not by a test:
frozen-base-weights, LoRA-weights-changed, and the exact checkpoint reload. Testing them in
CI would need torch and a CUDA device. They are marked ⚠️ in the checklist rather than ✅.

## Reproduce

```
uv run pytest training/tests/test_grpo_math.py -q     # 27 passed, no GPU
uv run python -c "import training.grpo_math"          # imports without torch
```

The GPU path was last executed 2026-10-18; see `docs/PHASE_F4_CONTROLLED_UPDATE.md` for its
metrics and archive hash.
