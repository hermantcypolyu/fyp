# Training readiness — the handoff doc

Written for someone with no memory of this build.

> **⚠️ This file did not exist until 2026-08-03.** It is referenced from nine places
> (`AGENT_BUILD_SPEC.md` §7, `README.md`, four ADRs, `PROGRESS.md`, `docs/nemo_gym_notes.md`,
> `docs/vram_notes.md`) and the spec calls it "the most important file in the repo", but it was
> never created. Those references were dangling. Recorded here rather than quietly fixed.

**➡️ To actually run anything, follow [`RUN_ON_A5000.md`](RUN_ON_A5000.md).** It is the ordered
runbook: verify the machine → clear the four blocking carry-forward items → install → verify the
port → baseline → 2-step smoke → full run. This document is the inventory and the reasoning
behind it.

---

## 1. What exists

**Benchmark — 36 tasks, all validated, all solvable by gold and failable without it.**

| Split | Tasks | bug_fix | config_fix | docs | feature | test_update | easy/med/hard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 26 | 10 | 4 | 3 | 6 | 3 | 8 / 13 / 5 |
| held-out | 10 | 4 | 1 | 1 | 2 | 2 | 4 / 4 / 2 |

- `real_repo` provenance: **17/36 (47%)** overall; **6/10 (60%)** held-out.
- Every task carries a validator-**measured** `n_tests`, `floor_tests` and `repo.loc`.
- Reward floor: train mean **0.46**, held-out mean **0.48** — the splits are well matched.
- Held-out headroom (`n_tests - floor_tests`) is **6–9 on every task**.

**Environment** — seven tools (`list_files`, `read_file`, `write_file`, `edit_file`,
`replace_lines`, `run_tests`, `submit`), a sandbox, and one verifier used by both evaluation
and RL (§5.6). Two transports:
`direct` (in-process) and `gym` (FastAPI over NeMo Gym).

**Reward** — `reward_v1`, normalised per ADR 0006:

```
reward = (passed - floor_tests) / (n_tests - floor_tests)    ∈ [-1, 1]
gates                                                        = -1.0
```

Doing nothing scores exactly **0.0**, gold **1.0**, breaking a previously-passing test **negative**.
Six hard gates: `tests_modified`, `no_submit`, `sandbox_escape`, `syntax_broken`, `timeout`,
`test_count_mismatch`.

**Tested:** 191 unit tests (1 skipped), 0 ruff findings, schema validation, gold-patch verification,
sandbox isolation, the reward invariants, and the held-out seal.

**Untested / never executed:** everything requiring a GPU or POSIX — see §2.

---

## 2. What was never run

- **No training.** Not one step, on any hardware.
- **No weights downloaded.** No model has ever been loaded.
- **No full evaluation sweep.** No baseline arm exists yet.
- **No held-out access.** The split is **SEALED** — see [`heldout_seal.md`](heldout_seal.md).
- **No GPU.** The build machine has no NVIDIA hardware at all (`nvidia-smi` is absent).
- **No `setrlimit` enforcement.** POSIX-only; all four caps report `False` on Windows.
- **No gym transport run here.** `uvloop` has no Windows build.

The four items that must be verified on Linux are enumerated in
[`A5000_CARRY_FORWARD.md`](A5000_CARRY_FORWARD.md) and are **blocking**.

---

## 3. Exact next commands

In order. Full detail and expected values in [`RUN_ON_A5000.md`](RUN_ON_A5000.md).

```bash
# 0. verify the box                       (needs the A5000)
nvidia-smi && python3 --version && uname -srm
uv run python scripts/capture_hardware.py --out docs/_hw_target.md

# 1. BLOCKING carry-forward, item 4 first (needs the A5000)
uv run pytest rewards/tests -q           # then prove the trainer does not clamp -1.0
uv sync --extra gym && make smoke-gym    # needs Linux
uv run python scripts/smoke_one_task.py --transport gym \
    --trace results/raw/b3_smoke/rollouts_gym.jsonl

# 2. verify the port                      (runs anywhere)
uv run python benchmark/schema/validate_tasks.py --all
uv run python scripts/verify_gold_patches.py --split all
make smoke
make check

# 3. baseline control arm                 (needs a GPU)
#    configs/eval/baseline.yaml — train split only

# 4. 2-step GRPO smoke                     (needs the A5000)
#    configs/training/grpo_unsloth_a5000.yaml with max_steps: 2

# 5. full run                              (needs the A5000)
```

Steps 0, 1, 3, 4, 5 need hardware this repository has never had. Step 2 runs anywhere and is
**MEASURED** on the build machine.

---

## 4. Hardware decision table

| Path | Hardware | Status | Notes |
| --- | --- | --- | --- |
| Baseline eval | A5000 24 GB | ⚠️ never run | `configs/eval/baseline.yaml`; 1.5B, 4-bit |
| QLoRA GRPO (Unsloth) | A5000 24 GB | ⚠️ never run | `configs/training/grpo_unsloth_a5000.yaml`; every value tagged GROUNDED / CHOSEN / ASSUMPTION |
| NeMo RL GRPO | 8× 80 GB node | ❌ blocked | per NVIDIA's documented prerequisites; also needs Python 3.13, a second interpreter |
| Full fine-tune | any single card | ❌ impossible | 16·P bytes ≈ 22.9 GiB of optimiser state alone for a 1.5B model, before activations |

**The target changed.** `AGENT_BUILD_SPEC.md` §2, `docs/vram_notes.md` and ADR 0001 all still say
**RTX 4070 Ti, 12 GB**. The training box is an **A5000, 24 GB**. The extra headroom is why
`max_seq_length: 8192` and `num_generations: 8` are plausible at all — on 12 GB they would not
have been. `vram_notes.md` §2–§3 must be re-derived on the real card.

---

## 5. Blockers & options

- **The A5000 is available**, so the single-GPU QLoRA path is the plan of record. This is a
  constraint, not a preference: full fine-tuning does not fit on any single consumer card.
- **NeMo RL's documented GRPO recipe needs 8× 80 GB.** That is out of reach. It is cloned and read
  for reference only — nothing in this repo imports it.
- If more compute becomes available: NVIDIA mentor compute request · university HPC · rented GPUs.
  None are required for the plan of record.

---

## 6. Known unknowns

Every one of these is an assumption that has never executed.

1. **The Unsloth × NeMo Gym integration is documented as _single-turn_ rollout RL.** Its reward
   function scores one completion with one `/verify` POST. **Our episode is multi-turn** —
   `seed_session`, up to `max_turns` tool calls, then `submit`. This is **not a drop-in**; a rollout
   adapter is needed. Specified in
   [ADR 0007](decisions/0007-multi-turn-rollout-adapter-and-unscorable-rollouts.md) and to be
   built **before any hyperparameter work**.
2. **Negative rewards through the trainer.** If anything clamps or relu's the reward, the `-1.0`
   gates collapse onto `0.0` and the reward-hack deterrent disappears silently. Carry-forward item 4.
3. **A failed `/verify` must not score.** Resolved in this repo by `rewards/transport.py`
   (`score_or_discard` returns `None`; `None` drops the rollout), but the adapter must honour it.
   Grouped with the other two in
   [Failures that look like valid scores](RUN_ON_A5000.md#failures-that-look-like-valid-scores).
4. **`max_seq_length: 8192`** — the upstream notebook uses 4096, measured on an H100 (80 GB).
5. **`gradient_accumulation_steps: 16`** — upstream uses 64; this changes effective batch size and
   learning dynamics.
6. **KV-cache arithmetic (~28 KiB/token)** in `vram_notes.md` is itself marked UNVERIFIED.
7. **Model substitution** — the notebook uses `Qwen2.5-1.5B-Instruct`; we use the `-Coder-` variant.
   Same architecture and parameter count, but unverified in this pipeline.
8. **`setrlimit` caps** have never constrained a real process.
9. **Both transports must agree on reward** (§5.6). Verified for `direct` only.

---

## 7. VRAM levers

Ordered; apply top-down on OOM. All **PROJECTED**. Full table with the arithmetic in
[`RUN_ON_A5000.md`](RUN_ON_A5000.md#vram-levers).

1. 4-bit base + LoRA — already applied, and not optional
2. `num_generations` 8 → 4 — halves rollout KV cache, but **halves the GRPO group**, so the
   advantage estimate gets noisier; a statistical cost, not just a speed one
3. `max_seq_length` 8192 → 4096 — back to the upstream value
4. `gradient_accumulation_steps` ↑ — VRAM-neutral, preserves effective batch
5. Tool-output truncation in the harness — caps prompt growth per turn
6. LoRA `r` 32 → 16 — smallest effect; do this last

---

## 8. Risk register

| Risk | Mitigation in place | Residual |
| --- | --- | --- |
| **Reward hacking** (`tests_modified` is the #1 hack) | Hidden tests rebuilt from git for every grade; six hard gates at `-1.0`; the denominator pinned to `task.yaml` so shrinking it cannot pay | A trainer-side clamp would silently disable it — carry-forward item 4 |
| **Overfitting to 26 train tasks** | Held-out split sealed and category/difficulty-matched; floors matched (0.46 vs 0.48) | 26 tasks is small; treat effect sizes with suspicion |
| **Contamination of held-out** | Sealed by tree hash, enforced by `scripts/check_heldout_seal.py` in `make check`; both failure routes tested | Only enforced where `make check` runs |
| **Train/held-out repo overlap** | Documented: `tenacity` and `funcy` are shared; `mistune` supplies 3 of 10 held-out tasks | Not fully independent; stated as a limitation |
| **Harness / API drift** | NeMo Gym and NeMo RL commits pinned; every trace records pinned vs actual with a `matches` flag | Both stacks are pre-1.0 and move fast |
| **Reward resolution** | Held-out headroom 6–9 everywhere; ADR 0006 normalisation puts the do-nothing plateau at 0.0 | Five train tasks have ≤5 movable tests |
| **Single-turn vs multi-turn mismatch** | Documented above and in the training config | Unresolved until the adapter is built |

---

## Related

- [`RUN_ON_A5000.md`](RUN_ON_A5000.md) — **the runbook; start here**
- [`A5000_CARRY_FORWARD.md`](A5000_CARRY_FORWARD.md) — the four blocking items
- [`heldout_seal.md`](heldout_seal.md) — what sealing means
- [`pinned_versions.md`](pinned_versions.md) — upstream commits, and the Python 3.12/3.13 split
- [`vram_notes.md`](vram_notes.md) — memory arithmetic (written for 12 GB)
- `../PROGRESS.md` — the full build log
