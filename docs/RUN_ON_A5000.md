# Running this repository on an RTX A5000

Ordered runbook for the first session on the training box. Follow it top to bottom;
each phase gates the next.

**Nothing below has been executed.** This repository was built on Windows, CPU-only, with
no GPU, no training run and no weights downloaded. Every figure is tagged:

- **MEASURED** — a command produced it on the build machine, output saved under `results/raw/`
- **PROJECTED** — arithmetic or an upstream claim, never verified here

---

## Phase 0 — verify the machine

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

## Phase 1 — BLOCKING: clear the carry-forward

`docs/A5000_CARRY_FORWARD.md` lists four things that have **never executed on any machine**.
Clear all four before anything else. **Item 4 first** — it is the one that can invalidate a
whole training run without producing an error.

### 1.4 (FIRST) — confirm the trainer does not clamp negative rewards

`reward_v1` is in `[-1, 1]`. Every hard gate returns exactly `-1.0`, and that is the only
thing making the #1 reward hack (`tests_modified`) worse than doing nothing.

**If anything in the trainer applies `max(0, r)`, a relu, a `[0,1]` normalisation or a
clip, the gates collapse onto `0.0` — which is what doing nothing scores — and the
deterrent disappears with no error and no failing test.**

```bash
# read the reward path end to end, then prove it with a gated rollout
uv run pytest rewards/tests -q          # the local invariant, already green
```

Then drive one deliberately gated episode (end without `submit`) through a single training
step and assert the advantage it receives is the most negative in its group. **If the
trainer clamps, fix the trainer or add a thin adapter. Never raise `GATE_REWARD`** — that
re-opens the hack this is all guarding.

### 1.1 — `ResourceLimits.enforced()`

Four `setrlimit` caps (wall clock, CPU, address space, file size) have **never constrained a
process**: `resource` is POSIX-only, so on Windows `preexec_fn()` returns `None` and all four
report `False`.

```bash
uv run pytest envs/tests -q
uv run python scripts/smoke_one_task.py --limit 1
```

Confirm `limits_enforced` is `true` for all four in the trace, then force each cap once and
check the episode ends as a **scored row with the `timeout` gate**, not a hang or a traceback.

### 1.2 — gym transport end to end

```bash
uv sync --extra gym
make smoke-gym
```

Reward must be **1.0** and must equal the `--transport direct` reward for the same task.
That equality is §5.6's one-verifier invariant and the most important thing to re-check
after ADR 0006.

### 1.3 — regenerate the stale gym trace

```bash
uv run python scripts/smoke_one_task.py --transport gym \
    --trace results/raw/b3_smoke/rollouts_gym.jsonl
```

`results/raw/b3_smoke/rollouts_gym.jsonl` is still **schema v1** from B3 and is the last v1
artefact in the repo. Confirm the new row reads `schema_version: "2"` and carries
`floor_tests`.

---

## Phase 2 — install

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

## Phase 3 — verify the port before trusting anything

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

## Phase 4 — baseline (the control arm)

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

## Phase 5 — 2-step GRPO smoke

```bash
# config: configs/training/grpo_unsloth_a5000.yaml, with max_steps: 2
```

The point is not learning. It is to prove the loop closes: rollouts generate, `/verify` is
reached, a scalar reward comes back, the advantage is finite, and a step completes without
OOM. Watch `nvidia-smi` throughout and record peak VRAM — that is the first real number this
project will have about memory.

> **Expect to write a rollout adapter here.** Unsloth's NeMo Gym integration is documented as
> **single-turn** rollout RL: its `reward_fn(completions, prompts, **kwargs)` scores one
> completion with one `/verify` POST. Our episode is multi-turn — `seed_session`, up to
> `max_turns` tool calls, then `submit`. This is **not** a drop-in. Prototype the adapter
> before touching hyperparameters.

> **Do not let a failed `/verify` score.** The upstream reward function returns `0.0` when the
> HTTP call fails or is non-200. Under ADR 0006 `0.0` is a real score — "achieved nothing" —
> so a transport failure would enter the dataset indistinguishable from a genuine do-nothing
> rollout. Make it raise, or record a sentinel and drop the row.

---

## Phase 6 — full run

Only after Phases 1–5 are green. Checkpoint selection uses a **validation slice carved from
train**. The held-out split is not touched, not inspected, and not used to choose anything.

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
