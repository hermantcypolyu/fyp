# A5000 carry-forward — what has never executed on this machine

Everything in this repository was built on **Windows**, against a CPU-only venv, with no
training run and no weights downloaded. Several things are therefore **specified, committed
and unit-tested, but have never actually run**. They are listed here so that the first
session on the A5000 box verifies them deliberately rather than discovering them mid-run.

Each item states what is unverified, why it could not run here, and the exact check to
perform on Linux.

## Status at a glance

| # | Item | Status |
| --- | --- | --- |
| 1 | `ResourceLimits.enforced()` — the caps are applied | ✅ verified 2026-08-03 |
| 1b | Each cap actually **stops** a hostile process | ⏳ **open** |
| 2 | gym transport, end to end | ✅ verified 2026-08-03 |
| 3 | `rollouts_gym.jsonl` at trace schema v2 | ✅ verified 2026-08-03 |
| 4 | Reward range `[-1, 1]` survives the trainer | ⏳ **open — highest risk** |

First Linux session: 2026-08-03, WSL2 Ubuntu on the A5000 box (24 GB, driver CUDA 13.0,
`torch.cuda.is_available()` → `True`). Item 2 required an unforeseen fix — see
[pinned_versions.md](pinned_versions.md) §"Pinning the source clone does not pin the imported
artifact" — which is the single most useful thing this document caused to be found.

---

## 1. `ResourceLimits.enforced()` — the caps had never run

**Status:** ✅ **verified on Linux 2026-08-03.** The applied/not-applied half is closed;
forcing each cap (item 1b) is still open.

`envs/coding_env/limits.py` applies POSIX resource limits to the graded pytest subprocess via
a `preexec_fn`. The `resource` module is POSIX-only, so on Windows `preexec_fn()` returns
`None` and `enforced()` reports every `setrlimit` cap as `False`. **No cap had ever actually
constrained a process.**

This matters because the caps are a containment control, not a convenience: a task that
allocates without bound, forks, or writes a huge file is supposed to be stopped by the
kernel, and a `timeout` gate is supposed to be a *scored failure* rather than a hung run.

### Correction: there are six caps, not four

The original text here said "four `setrlimit` caps". The trace records **six** keys, of which
four are `setrlimit` and two were already cross-platform. The Windows→Linux delta is exactly
the four POSIX ones:

| Cap | Mechanism | Windows | Linux |
| --- | --- | --- | --- |
| `cpu_s` | `setrlimit` | ❌ `false` | ✅ `true` |
| `memory_mb` | `setrlimit` (address space) | ❌ `false` | ✅ `true` |
| `max_file_mb` | `setrlimit` (file size) | ❌ `false` | ✅ `true` |
| `max_processes` | `setrlimit` | ❌ `false` | ✅ `true` |
| `wall_clock_s` | subprocess timeout | ✅ `true` | ✅ `true` |
| `network` | sandbox by absence | ✅ `true` | ✅ `true` |

Verified from a trace written on the box:

```json
{"cpu_s": true, "max_file_mb": true, "max_processes": true,
 "memory_mb": true, "network": true, "wall_clock_s": true}
```

Both `results/raw/b3_smoke/rollouts_direct.jsonl` and `rollouts_gym.jsonl` were regenerated on
Linux so that no artefact in `results/` still carries the Windows all-`false` record. The
Windows originals are kept as `*.bak` (gitignored) for the before/after comparison.

**Commands run:**

```bash
uv run pytest envs/tests -q -rs                  # 101 passed, 1 skipped (the skip was item 2)
uv run python scripts/smoke_one_task.py --limit 1
uv run python scripts/smoke_one_task.py --limit 1 --trace results/raw/linux_verify/limits_probe.jsonl
```

Note: `smoke_one_task.py` writes a trace **only** when `--trace` is passed. Without it the
command prints a reward and inspecting an older trace file measures nothing.

### 1b. Forcing each cap — still open

`limits_enforced: true` proves the caps are *set*, not that they *bite*. Each still needs to
be driven past its limit at least once: a deliberate infinite loop (CPU), a large allocation
(address space), a large write (file size), and a sleep past the wall clock — checking that
the episode ends with the `timeout` gate and a **scored row** rather than an exception or a
hang.

Keep the probes out of `benchmark/tasks/` so they never reach `make check` or the corpus
statistics; `scratch/` is excluded via `.git/info/exclude`.

---

## 2. gym transport, end to end

**Status:** ✅ **verified on Linux 2026-08-03.** The §5.6 invariant holds.

`nemo-gym` depends on `uvloop`, which has no Windows build (ADR 0002), so the `gym` extra
does not install there and `envs/coding_env/app.py` cannot even be imported. Its structure is
asserted statically with `ast` in `envs/tests/test_server_contract.py`, including — since
ADR 0006 — that `CodingVerifyResponse` declares `floor_tests` and that `app.py` populates it.
Static assertions are not a substitute for running it.

### It did not work first time

`uv sync --extra gym` installed the PyPI wheel `nemo-gym==0.4.0`, which predates
`ReverifyMode`. `app.py:36` imports it, so `--transport gym` died on an `ImportError` while
`analysis/versions.py` still reported `matches: true` for the clone. The clone was pinned; the
*imported artifact* was not. Fixed by installing the vendored clone (`0.5.0rc0`) over the
wheel; the mechanism, the trap and the `make gym-vendor` target are documented in
[pinned_versions.md](pinned_versions.md).

**Commands run:**

```bash
make gym-vendor          # uv sync --extra gym, then the vendored editable install
make smoke-gym
```

### Result

| Transport | Reward | Tests | Gates |
| --- | --- | --- | --- |
| `direct` | **1.00** | 12/12 | — |
| `gym` | **1.00** | 12/12 | — |

Task `bugfix_dotenv_bom_prefix`, same episode, same number. That equality is §5.6's invariant
— one verifier for both evaluation and RL reward — and it was the single most important thing
to re-check after the reward change.

Side effect: `envs/tests/test_server_contract.py:159` stopped skipping, and the full suite
went from **230 passed / 2 skipped** to **232 passed / 0 skipped**. No test in this repository
is skipped on Linux.

---

## 3. Regenerate `rollouts_gym.jsonl` at trace schema v2

**Status:** ✅ **verified on Linux 2026-08-03.** No v1 artefact remains.

ADR 0006 bumped the trace schema to v2: `reward` is now `[-1, 1]` and `result.floor_tests` is
recorded per rollout. `results/raw/b3_smoke/rollouts_gym.jsonl` predated that, so it had no
`floor_tests` field and declared `schema_version: "1"`.

The stale file was not *wrong* — a gold rollout scores 1.0 under both the old and the new
formula, so its headline number still held — but it could not be mixed with v2 rows in any
aggregate, and it was the only v1 artefact left.

**Command run:**

```bash
uv run python scripts/smoke_one_task.py --transport gym \
    --trace results/raw/b3_smoke/rollouts_gym.jsonl
```

### Result, and an unplanned cross-check

The row reads `schema_version: "2"` and carries `floor_tests: 7`. That number is worth
pausing on. `scripts/verify_gold_patches.py` measures the same task statically:

```
bugfix_dotenv_bom_prefix   untouched 7/12 passed   +gold 12/12 passed   floor 0.58
```

7/12 = 0.583. The floor recorded at runtime in the trace is the same floor measured at
benchmark-build time — the ADR 0006 denominator survives the whole path from `task.yaml`
through `/verify` into the trace unchanged. That is an end-to-end check of the reward
pipeline that neither script was written to perform.

`rollouts_direct.jsonl` was regenerated at the same time (see item 1), so both artefacts are
now Linux-written, schema v2, and carry six enforced caps.

---

## 4. Confirm the v2 reward range `[-1, 1]` suits the trainer's advantage computation

**Status:** ⏳ **open, and the highest-risk item here.** No trainer has ever consumed a reward
from this environment.

ADR 0006 widened `reward_v1` from `[0, 1]` to `[-1, 1]`: a rollout that breaks tests which
passed untouched now scores negative, and every hard gate scores exactly `-1.0`. That is
deliberate — it is what makes "made it worse" distinguishable from "did nothing", and what
keeps the #1 reward hack strictly worse than an immediate submit.

**Some implementations assume non-negative rewards.** Failure modes to rule out before any
real run, none of which necessarily raise an error:

- a reward clamp or `relu`/`max(0, r)` anywhere in the trainer's reward pipeline, which would
  silently collapse `-1.0` gates and negative regressions onto the same `0.0` as doing
  nothing — reinstating exactly the hazard ADR 0006's coupling constraint exists to prevent;
- reward normalisation that assumes a `[0, 1]` domain, e.g. dividing by a running max;
- return/advantage whitening that special-cases sign, or a value head initialised on the
  assumption of non-negative returns;
- logging or dashboards that clip at zero, which would hide the problem rather than cause it.

GRPO's own advantage, $(r-\mu_g)/\sigma_g$, is sign-agnostic and needs no change; this is a
question about the surrounding implementation, not the algorithm. `rewards/tests/
test_compose.py` already checks the arithmetic locally — a mixed group `[0.0, 0.5, 1.0, -1.0]`
has non-zero σ, produces no NaN, and puts the gated rollout at the most negative advantage.
Re-run on Linux 2026-08-03: **16 passed**. The local half of this item is sound; what remains
is the trainer's half.

This is **hazard 2 of three** that share a failure signature. See
[Failures that look like valid scores](RUN_ON_A5000.md#failures-that-look-like-valid-scores) for
the full set — the third is a failed `/verify` being scored as `0.0`, closed in this repo by
`rewards/transport.py` and [ADR 0007](decisions/0007-multi-turn-rollout-adapter-and-unscorable-rollouts.md).

**Verify on Linux, before training:** read the trainer's reward path end to end and confirm a
`-1.0` survives from `verify()` to the advantage unchanged. Start with a source audit of the
pinned clone:

```bash
grep -rnE "clamp|torch\.clip|relu\(|max\(0" ../_vendor/nemo-rl/nemo_rl/algorithms/
grep -rnE "advantage|baseline|whiten|normalize" ../_vendor/nemo-rl/nemo_rl/algorithms/grpo.py
grep -rn "reward" ../_vendor/nemo-rl/nemo_rl/environments/
```

(`grep` needs `-E` for alternation; without it the pattern is matched literally and an empty
result means nothing.)

A cheap direct check is to inject a deliberately gated rollout (end an episode without
`submit`) into a single-step run and assert the advantage it receives is the most negative in
its group. If the trainer does clamp, the fix is in the trainer's config or a thin adapter —
**not** by raising `GATE_REWARD`, which would re-open the hack.

---

## Also unresolved, carried from the closing commit

Neither of these is a "never executed" item, but both block the same milestone:

- **Trainer selection** (`16f3901`). Which NeMo RL entry point and config the training phase
  uses is not decided.
- **Single-turn vs multi-turn mismatch** — TRAINING_READINESS records this as *"unresolved
  until the adapter is built"*. ADR 0007 specifies the rollout adapter; it has not been built.

---

## Not carry-forward, but worth stating

- **No training has been run**, no weights downloaded, and no checkpoint exists. Every number
  recorded in `PROGRESS.md` is a benchmark or environment measurement.
  *(Updated 2026-08-03: `Qwen2.5-Coder-1.5B-Instruct` (2.9 GB) and `Qwen2.5-Coder-3B-Instruct`
  (5.8 GB) are now cached on the box. Still no training run and no checkpoint.)*
- **The held-out split is sealed** (`docs/heldout_seal.md`). The first A5000 session must not
  touch it; `make check` enforces this. Confirmed intact on 2026-08-03: tree `318fedc8868b`
  unchanged, 10 tasks.
- Two train tasks exceed the 300 LOC guidance (346 and 376) and are accepted as a stated
  limitation — see `PROGRESS.md`, 2026-08-03 Step 1.
- `make check` on Linux reports the same five warnings as on Windows — two `LOC_BIG`, the
  `test_update` 3/6 and `config_fix` 4/5 mix shortfalls, and `real_repo` provenance at 47%
  against a ≥50% target. All were known before the move; none are new.
- `vram_notes.md` was written for a 12 GB budget. The box has **24 GB**. The arithmetic there
  is a safe lower bound rather than the binding constraint, but it has not been redone.