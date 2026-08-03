# Running this repository on an RTX A5000

Ordered runbook for the first session on the training box. Follow it top to bottom;
each phase gates the next.

**Nothing below has been executed.** This repository was built on Windows, CPU-only, with
no GPU, no training run and no weights downloaded. Every figure is tagged:

- **MEASURED** — a command produced it on the build machine, output saved under `results/raw/`
- **PROJECTED** — arithmetic or an upstream claim, never verified here

---

## Phase 0 — FIRST, before you run anything: inspect the trainer for a reward clamp

This is a **code inspection**. It needs no GPU, no install and no model. Do it before
`nvidia-smi`, because it decides whether any run you do afterwards is worth keeping.

`reward_v1` is in `[-1, 1]`. Every hard gate returns exactly `-1.0`, and that is the only thing
making the #1 reward hack (`tests_modified`) worse than doing nothing. Read the trainer's reward
path end to end and find every place a reward is transformed between `verify()` and the advantage:

```bash
# in whichever trainer you land on (see "Trainer selection is UNRESOLVED" below)
grep -rnE 'max\(0|relu|clamp|clip|\.abs\(|normali[sz]e' <trainer>/  # start here, then read properly
```

Look for, and rule out:

- `max(0, r)`, `relu`, `torch.clamp(min=0)`, `np.clip(r, 0, 1)`
- reward normalisation assuming a `[0, 1]` domain (e.g. dividing by a running max)
- return/advantage whitening that special-cases sign
- a value head or baseline initialised assuming non-negative returns
- logging or dashboards that clip at zero — these hide it rather than cause it

**If anything clamps, the `-1.0` gates collapse onto `0.0` — which is exactly what doing nothing
scores — and the deterrent disappears with no error and no failing test.**

**The fix is in the trainer, or in a thin adapter around it. Never by raising `GATE_REWARD`** —
that re-opens the hack from the other side, and it is the one change that would look like a fix
while removing the protection entirely.

Once a trainer is chosen and read, prove it empirically: drive one deliberately gated episode (end
without `submit`) through a single training step and assert the advantage it receives is the most
negative in its group.

```bash
uv run pytest rewards/tests -q          # our own invariants — already green, not sufficient
```

Those tests prove *our* arithmetic. They cannot see inside the trainer, which is why this phase is
manual and comes first. See [Failures that look like valid scores](#failures-that-look-like-valid-scores).

---

## Phase 1 — verify the machine

```bash
nvidia-smi
python3 --version
uname -srm
```

| Check | Expect | Status |
| --- | --- | --- |
| GPU | `NVIDIA RTX A5000` | PROJECTED |
| VRAM total | **24576 MiB** | PROJECTED |
| Compute capability | 8.6 (Ampere) | PROJECTED |
| Python | **3.12.x** | required — NeMo Gym needs 3.12+, and `ray` publishes no wheel past cp312 |
| OS | Linux | required — `uvloop` and `setrlimit` are POSIX-only |

Then record the real numbers rather than trusting this table:

```bash
uv run python scripts/capture_hardware.py --out docs/_hw_target.md
```

> **VRAM free at idle matters more than VRAM total.** A headless box gives nearly all
> 24 GiB; one running a display holds back 0.5–1.5 GiB. Budget against measured free.

> **Two interpreters, if NeMo RL is ever used.** NeMo Gym runs on 3.12; NeMo RL requires
> `>=3.13.13,<3.14` and will refuse to install alongside it. This runbook uses Unsloth, not
> NeMo RL, so 3.12 alone is enough. See `docs/pinned_versions.md`.

---

## Phase 2 — BLOCKING: clear the rest of the carry-forward

`docs/A5000_CARRY_FORWARD.md` lists four things that have **never executed on any machine**.
Item 4 is Phase 0 above. Clear the remaining three before installing anything heavier.

### 2.1 — `ResourceLimits.enforced()`

Four `setrlimit` caps (wall clock, CPU, address space, file size) have **never constrained a
process**: `resource` is POSIX-only, so on Windows `preexec_fn()` returns `None` and all four
report `False`.

```bash
uv run pytest envs/tests -q
uv run python scripts/smoke_one_task.py --limit 1
```

Confirm `limits_enforced` is `true` for all four in the trace, then force each cap once and
check the episode ends as a **scored row with the `timeout` gate**, not a hang or a traceback.

### 2.2 — gym transport end to end

```bash
uv sync --extra gym
make smoke-gym
```

Reward must be **1.0** and must equal the `--transport direct` reward for the same task.
That equality is §5.6's one-verifier invariant and the most important thing to re-check
after ADR 0006.

### 2.3 — regenerate the stale gym trace

```bash
uv run python scripts/smoke_one_task.py --transport gym \
    --trace results/raw/b3_smoke/rollouts_gym.jsonl
```

`results/raw/b3_smoke/rollouts_gym.jsonl` is still **schema v1** from B3 and is the last v1
artefact in the repo. Confirm the new row reads `schema_version: "2"` and carries
`floor_tests`.

---

## Phase 3 — install

`pip` may not be present. Everything goes through `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # if uv is absent
git clone <this repo> && cd rl-coding-agents

uv sync                       # core + dev, Python 3.12
uv sync --extra gym           # NeMo Gym stack (Linux only)
```

Unsloth is **not** a dependency of this repo and is installed separately, into its own
environment, pinned to the versions NeMo Gym is tested against:

```bash
uv pip install "unsloth==2026.1.4" "unsloth_zoo==2026.1.4" omegaconf
```

> Those two pins are not a preference. NeMo Gym's compatibility reference states the
> integration is tested on exactly `unsloth==2026.1.4` / `unsloth_zoo==2026.1.4` and that
> "other versions are not guaranteed to work".

---

## Phase 4 — verify the port before trusting anything

Run in this order. Each is cheap and each catches a different class of breakage.

```bash
uv run python benchmark/schema/validate_tasks.py --all     # 36 tasks, schema + counts
uv run python scripts/verify_gold_patches.py --split all   # every gold patch still solves
make smoke                                                  # one full episode, real tools
make check                                                  # includes the held-out seal check
```

| Expected | Value | Status |
| --- | --- | --- |
| Validator | 36 tasks, 0 errors, 5 warnings | **MEASURED** (build machine) |
| Gold patches | 36/36 solvable and failable | **MEASURED** |
| Reward floor, train | min 0.00 / mean 0.46 / max 0.58 | **MEASURED** |
| Reward floor, held-out | min 0.38 / mean 0.48 / max 0.62 | **MEASURED** |
| Unit tests | 191 passed, 1 skipped | **MEASURED** |
| `make smoke` reward | 1.0 | **MEASURED** (direct transport only) |
| Held-out seal | tree `318fedc8868b` unchanged | **MEASURED** |

The five warnings are known and accepted: two `LOC_BIG` (346 and 376 LOC train tasks), two
corpus-mix shortfalls, one provenance figure at 47%. See `PROGRESS.md`.

---

## Phase 5 — baseline (the control arm)

Do this **before** any training. Without it there is nothing to compare against.

```bash
# config: configs/eval/baseline.yaml
# train split only — held-out is sealed
```

Records `reward`, `pass_fraction`, `floor_tests`, per-gate rates and `tool_calls` per rollout
to `results/raw/b6_baseline/`. **Report the gate rate beside the reward**: a mean of 0.3 with
a 40% `tests_modified` rate is a different result from 0.3 with 0%.

PROJECTED: a 1.5B instruct model with no RL should do poorly at multi-turn repair — expect
many `no_submit` gates and a mean reward near **0.0**, which under ADR 0006 is exactly what
"achieved nothing" scores. A near-zero baseline is the expectation, not a bug.

---

## Phase 6 — 2-step GRPO smoke

```bash
# config: configs/training/grpo_unsloth_a5000.yaml, with max_steps: 2
```

The point is not learning. It is to prove the loop closes: rollouts generate, `/verify` is
reached, a scalar reward comes back, the advantage is finite, and a step completes without
OOM. Watch `nvidia-smi` throughout and record peak VRAM — that is the first real number this
project will have about memory.

> **Expect to write a rollout adapter here — do it before any hyperparameter work.** Unsloth's
> NeMo Gym integration is documented as **single-turn** rollout RL: its
> `reward_fn(completions, prompts, **kwargs)` scores one completion with one `/verify` POST. Our
> episode is multi-turn — `seed_session`, up to `max_turns` tool calls, then `submit`. This is
> **not** a drop-in. The design, including how token ids and logprobs survive the adapter and
> which turns receive a gradient, is specified in
> [ADR 0007](decisions/0007-multi-turn-rollout-adapter-and-unscorable-rollouts.md).

> **A failed `/verify` must not score.** Use `rewards.transport.score_or_discard`; it returns
> `None` on any transport failure, and `None` means *drop the rollout from the group*. Log
> `DiscardTracker.summary()` every step and call `check()`. See
> [Failures that look like valid scores](#failures-that-look-like-valid-scores).

---

## Phase 7 — full run

Only after Phases 0–6 are green. Checkpoint selection uses a **validation slice carved from
train**. The held-out split is not touched, not inspected, and not used to choose anything.

---

## Failures that look like valid scores

Three hazards share one signature: **the run keeps going, the loss curve looks plausible, and
nothing errors** — while the number being optimised has quietly stopped meaning what it says.
None of them is caught by a test that has ever gone red on this machine, so each has to be
checked deliberately.

They are grouped here because they are the same bug wearing three hats: **a value that is not a
score is allowed to enter the reward stream as if it were one.**

| # | Hazard | What collapses | Status |
| --- | --- | --- | --- |
| 1 | Gate constant left at `0.0` while the reward is normalised | `tests_modified` scores the same as an immediate submit — the #1 reward hack becomes free | **closed by design** — `GATE_REWARD` is derived from `REWARD_MIN` in `rewards/compose.py`, pinned by `test_a_gated_rollout_scores_below_doing_nothing` |
| 2 | Trainer clamps or applies `max(0, r)` | every `-1.0` gate lands on `0.0` == "did nothing"; hazard 1 returns from outside our code | **OPEN — blocking. This is [Phase 0](#phase-0--first-before-you-run-anything-inspect-the-trainer-for-a-reward-clamp), the first action on the box.** Carry-forward item 4 |
| 3 | Failed `/verify` scored as `0.0` | counterfeit do-nothing rollouts enter the group, shifting the mean and σ that GRPO normalises against | **closed in our code** — `rewards/transport.py`; must be honoured by the adapter |
| 4 | Placeholder turn substituted for an empty rollout | a fabricated row with `generation_token_ids=[0]` is *trained on*; the loss mask is effectively zeroed and the sample teaches nothing while counting as a sample | **OPEN** — upstream behaviour; our adapter must discard instead |

**Hazard 4 in detail.** NeMo Gym's `verifiers_agent` guards against a rollout that produced no
trainable tokens — and then, rather than dropping it, appends a placeholder:

```python
if not any(item.get("generation_token_ids") for item in output):
    logger.warning("[verifiers_agent] rollout produced no trainable tokens. ... "
                   "Returning empty trajectory. Underlying error: %s", err_msg)
    output.append(NeMoGymResponseOutputMessageForTraining(
        id="msg_empty", content=[...text=""],
        prompt_token_ids=[0], generation_token_ids=[0], generation_log_probs=[0.0],
    ).model_dump())
```

The comment names the real cause — "this can happen when sandbox concurrency quota is exceeded" —
which is precisely the condition a long agentic rollout hits under load. It logs a **warning**, not
an error, and the run continues with a row that looks trainable and is not. TRL's vLLM model server
tests describe the same failure from the other end: emptied token ids "zeroing the training loss
mask".

Our adapter must treat "no trainable tokens" the same way it treats a failed `/verify`: **discard
the rollout and count it**, via `rewards.transport.DiscardTracker`. A rollout with no tokens is
unscorable by definition — there is nothing to attribute the reward to.

**Why `0.0` is the common thread.** Before ADR 0006, `0.0` was a safe fallback: it sat below every
task's floor, so anything that produced it was obviously bad. After normalisation `0.0` is a
*real, correct, common* score — exactly what a do-nothing rollout earns. Every place that used to
default to zero is now a place that fabricates a plausible sample.

Both reference implementations do exactly this, which is why hazard 3 is not hypothetical:

```python
# unslothai/notebooks, NeMo-Gym-Multi-Environment.ipynb
reward = resp.json().get("reward", 0.0) if resp.status_code == 200 else 0.0

# NVIDIA-NeMo/Gym, responses_api_agents/verifiers_agent/app.py
reward = rollout_output.get("reward", 0.0) or 0.0
```

**What to do about each**

- **Hazard 2** — read the trainer's reward path end to end before the first real run. Inject a
  deliberately gated rollout (end without `submit`) and assert its advantage is the most negative
  in its group. **If the trainer clamps, fix the trainer. Never raise `GATE_REWARD`** — that
  re-opens hazard 1 from the other direction.
- **Hazard 3** — the adapter must call `rewards.transport.score_or_discard`, which returns `None`
  on any transport failure. `None` means **drop the rollout from the group**, TRL's own contract
  for an unscorable rollout. Log `DiscardTracker.summary()` every step and call `check()`; a
  discard rate above 2% stops the run, because the rollouts that did score are then a biased
  sample of a dying environment.
- **Hazard 1** — already closed, but `make check` must stay green; the coupling test is what keeps
  it closed.
- **Hazard 4** — in the adapter, check for trainable tokens *before* scoring. No tokens → discard,
  do not substitute a placeholder.

See [ADR 0007](decisions/0007-multi-turn-rollout-adapter-and-unscorable-rollouts.md).

---

## Trainer selection is UNRESOLVED

**This is the prototype's first question**, and it must be answered before hyperparameters are
touched. Two candidate paths exist and neither is obviously right.

| | **TRL loop-owning harness** | **Unsloth × NeMo Gym** |
| --- | --- | --- |
| Entry point | `experimental.async_grpo.AsyncGRPOTrainer` + `HarnessRolloutWorker` | `trl.GRPOTrainer` per the NeMo Gym tutorial |
| Turn model | multi-turn; the agent owns its own loop | **single-turn** rollout RL |
| Token ids / logprobs | captured by a transparent proxy as `TraceEntry.completion_token_ids` / `per_token_logps` | generated internally by the trainer |
| Unscorable rollouts | first-class: `rollout_reward_fn -> float \| None`, `None` drops from the group | not addressed; the notebook defaults to `0.0` |
| Turn masking | `train_turn_fn`, ships `has_tool_call` | none for interleaved tool output |
| Worked example nearest ours | `opencode` — a real coding agent scored by "the fraction of held-out tests that passed" | Sudoku / instruction-following |
| Memory optimisations | standard TRL | Unsloth's 4-bit + LoRA path, which is why the A5000 config exists |
| Stability | `trl.experimental` — "its API may change" | documented and version-pinned (`unsloth==2026.1.4`) |

**The tension.** The TRL harness path fits our architecture almost exactly — multi-turn, per-turn
token capture, a `None` sentinel, turn masking, and a coding-agent example with a held-out-test
reward. But it is experimental, and it is **not** the path Unsloth documents. Unsloth is where the
4-bit + LoRA memory story comes from, and that story is what makes a 24 GB card viable at all.
**Whether Unsloth's optimisations compose with `AsyncGRPOTrainer` is unverified.**

### Decision criteria, in order

1. **Does the reward survive intact?** A `-1.0` must reach the advantage as `-1.0`
   ([Phase 0](#phase-0--first-before-you-run-anything-inspect-the-trainer-for-a-reward-clamp)).
   A path that clamps is disqualified unless the clamp can be removed cleanly.
2. **Can an unscorable rollout be dropped rather than scored?** TRL's harness path has this
   natively. If the chosen path cannot express "no score", it needs an adapter that can, because
   hazards 3 and 4 have no other honest resolution.
3. **Are the sampled token ids and logprobs available per turn?** This is not negotiable — see
   ADR 0007. Re-tokenising a rendered transcript is **unsound, not untidy**: it is not the
   sequence the policy emitted, so the ratio is computed against a different distribution.
   Whatever trainer wins, this constraint holds.
4. **Can tool output be excluded from the gradient?** Tool results are environment text. Training
   on them teaches the model to predict its own tools' output.
5. **Does it fit in 24 GB with `num_generations: 8`?** Decided by measurement in Phase 6, not by
   argument.
6. **Only then:** speed and API stability.

Criteria 1–4 are correctness. Criterion 5 is feasibility. **If the two paths split on 1–4 and 5,
prefer correctness and buy memory back with the [VRAM levers](#vram-levers)** — a fast run whose
reward signal is subtly wrong produces a dissertation result that cannot be defended.

Record the outcome as an ADR before writing training code.

---

## VRAM levers

All **PROJECTED**. Apply top-down when you hit OOM.

| # | Lever | Effect | Cost |
| --- | --- | --- | --- |
| 1 | 4-bit base + LoRA | already in the config | full fine-tune is arithmetically impossible; not a choice |
| 2 | `num_generations` 8 → 4 | halves rollout KV cache | **halves the GRPO group** — noisier advantage, a statistical cost, not just speed |
| 3 | `max_seq_length` 8192 → 4096 | roughly halves KV cache and activations | truncates long episodes; 4096 is the value the upstream notebook actually uses |
| 4 | `gradient_accumulation_steps` ↑ | VRAM-neutral, preserves effective batch | slower wall-clock per step |
| 5 | Tool-output truncation in the harness | caps prompt growth per turn | the agent sees less of a long file |
| 6 | LoRA `r` 32 → 16 | smallest effect of the six | LoRA params are ~1% of the model; do this last |

Arithmetic behind lever 3, from `docs/vram_notes.md` §3.2 (itself marked UNVERIFIED there):
KV cache ≈ **28 KiB/token** for this architecture, so 8192 tokens × 8 generations ≈ **1.75 GiB**
of KV cache alone — before weights, activations or optimiser state.

> The 24 GiB card is roomy compared with the 12 GiB the spec originally assumed, so the
> expectation is that 8192 × 8 fits. That expectation is **PROJECTED and untested**; treat
> the first OOM as information, not failure.

---

## Related

- `docs/A5000_CARRY_FORWARD.md` — the four blocking items, in full
- `docs/heldout_seal.md` — what sealing means and how it is enforced
- `docs/pinned_versions.md` — NeMo Gym / NeMo RL commits and the Python split
- `docs/vram_notes.md` — memory arithmetic (written for 12 GB; re-derive for 24 GB)
- `configs/training/grpo_unsloth_a5000.yaml` — every value tagged GROUNDED / CHOSEN / ASSUMPTION
- `configs/eval/baseline.yaml` — the control arm
