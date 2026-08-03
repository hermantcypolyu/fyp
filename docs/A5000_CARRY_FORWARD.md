# A5000 carry-forward — what has never executed on this machine

Everything in this repository was built on **Windows**, against a CPU-only venv, with no
training run and no weights downloaded. Several things are therefore **specified, committed
and unit-tested, but have never actually run**. They are listed here so that the first
session on the A5000 box verifies them deliberately rather than discovering them mid-run.

Each item states what is unverified, why it could not run here, and the exact check to
perform on Linux.

---

## 1. `ResourceLimits.enforced()` — four `setrlimit` caps have never run

**Status:** never executed. The code path exists and is wired into every graded run.

`envs/coding_env/limits.py` applies four POSIX resource limits to the graded pytest
subprocess via a `preexec_fn`: wall clock, CPU seconds, address space, and file size. The
`resource` module is POSIX-only, so on Windows `preexec_fn()` returns `None` and
`enforced()` reports every cap as `False`. **No cap has ever actually constrained a process.**

This matters because the caps are a containment control, not a convenience: a task that
allocates without bound, forks, or writes a huge file is supposed to be stopped by the
kernel, and a `timeout` gate is supposed to be a *scored failure* rather than a hung run.

**Verify on Linux:**

```bash
uv run pytest envs/tests -q                     # limits tests stop skipping
uv run python scripts/smoke_one_task.py --limit 1
```

Then confirm in the trace that `limits_enforced` is `true` for all four caps, and force each
one at least once — a deliberate infinite loop (CPU), a large allocation (address space), a
large write (file size), and a sleep past the wall clock — checking that the episode ends with
the `timeout` gate and a scored row rather than an exception or a hang.

---

## 2. gym transport, end to end

**Status:** never executed on this machine. Last run in B3, on a different machine, under the
pre-ADR-0006 reward.

`nemo-gym` depends on `uvloop`, which has no Windows build (ADR 0002), so the `gym` extra
does not install here and `envs/coding_env/app.py` cannot even be imported. Its structure is
asserted statically with `ast` in `envs/tests/test_server_contract.py`, including — since
ADR 0006 — that `CodingVerifyResponse` declares `floor_tests` and that `app.py` populates it.
Static assertions are not a substitute for running it.

**Verify on Linux:**

```bash
uv sync --extra gym
make smoke-gym
```

The reward must be **1.0**, and must equal the `--transport direct` reward for the same task.
That equality is §5.6's invariant — one verifier for both evaluation and RL reward — and it is
the single most important thing to re-check after the reward change.

---

## 3. Regenerate `rollouts_gym.jsonl` at trace schema v2

**Status:** stale. The file on disk is **schema v1**, written in B3 before ADR 0006.

ADR 0006 bumped the trace schema to v2: `reward` is now `[-1, 1]` and `result.floor_tests` is
recorded per rollout. `results/raw/b3_smoke/rollouts_gym.jsonl` predates that, so it has no
`floor_tests` field and declares `schema_version: "1"`. `rollouts_direct.jsonl` has already
been regenerated at v2 on this machine.

The stale file is not *wrong* — a gold rollout scores 1.0 under both the old and the new
formula, so its headline number still holds — but it cannot be mixed with v2 rows in any
aggregate, and it is the only v1 artefact left.

**Verify on Linux:** after item 2 succeeds,

```bash
uv run python scripts/smoke_one_task.py --transport gym \
    --trace results/raw/b3_smoke/rollouts_gym.jsonl
```

Then confirm the row reads `schema_version: "2"` and carries `floor_tests`.

---

## 4. Confirm the v2 reward range `[-1, 1]` suits the trainer's advantage computation

**Status:** unverified, and the highest-risk item here. No trainer has ever consumed a reward
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

**Verify on Linux, before training:** read the trainer's reward path end to end and confirm a
`-1.0` survives from `verify()` to the advantage unchanged. A cheap direct check is to inject
a deliberately gated rollout (end an episode without `submit`) into a single-step run and
assert the advantage it receives is the most negative in its group. If the trainer does clamp,
the fix is in the trainer's config or a thin adapter — **not** by raising `GATE_REWARD`, which
would re-open the hack.

---

## Not carry-forward, but worth stating

- **No training has been run**, no weights downloaded, and no checkpoint exists. Every number
  recorded in `PROGRESS.md` is a benchmark or environment measurement.
- **The held-out split is sealed** (`docs/heldout_seal.md`). The first A5000 session must not
  touch it; `make check` enforces this.
- Two train tasks exceed the 300 LOC guidance (346 and 376) and are accepted as a stated
  limitation — see `PROGRESS.md`, 2026-08-03 Step 1.
