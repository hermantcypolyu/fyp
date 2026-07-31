# 0002 — Keep core dependencies minimal; NeMo Gym is an optional extra

- **Status:** Accepted
- **Date:** 2026-07-31
- **Milestone:** B0

## Context

`AGENT_BUILD_SPEC.md` §0.3 sets the acceptance bar: a person clones the repo on a *fresh machine*
and within ~30 minutes gets `uv sync` → `make validate` → `make smoke` working.

NeMo Gym (`nemo-gym==0.5.0rc0`, from `github.com/NVIDIA-NeMo/Gym` @ `ec5a8136`) declares 30+
runtime dependencies including `ray[default]>=2.56.1`, `mlflow>=3.14.0`, `datasets`, `wandb`,
`opensandbox`, `openshell`, `daytona` and `boto3`. Measured facts from this build:

| Fact | Evidence |
|---|---|
| `nemo_gym` fails to import on native Windows | `nemo_gym/server_utils.py:18` does an unconditional `import resource` (POSIX-only) |
| `uvloop` (a declared dep) has no Windows build | resolution error until overridden |
| `ray` publishes `win_amd64` wheels only up to **cp312** | PyPI JSON for `ray==2.56.1`: cp310/311/312 have `win_amd64`, cp313/cp314 do not |
| NeMo Gym's own README lists Windows as supported **only via WSL2** | `README.md` → Requirements table |

If `nemo-gym` were a hard dependency, `uv sync` would fail outright for a large fraction of
readers — including the person marking this project.

## Options considered

1. **Hard-depend on `nemo-gym`.** Truest to §5.6, but breaks §0.3 on Windows and on any Python
   ≠ 3.12 box, and makes the benchmark unusable without a 1 GB install.
2. **Vendor a copy of NeMo Gym into the repo.** Rejected: §1.5 forbids large blobs, it violates
   §9.5's spirit (we would drift from real APIs), and it creates a licence-attribution burden.
3. **Split the code: pure-Python core + thin NeMo Gym adapter, with `nemo-gym` behind an
   optional extra.** ✅ Chosen.

## Decision

```
[project]
dependencies = ["pyyaml", "jsonschema", "numpy"]      # core: OS-independent, no GPU, no Ray

[project.optional-dependencies]
gym = ["nemo-gym>=0.4.0"]                              # the HTTP server stack (Linux/macOS/WSL2)
dev = ["pytest", "pytest-cov", "ruff"]
```

Consequently `envs/coding_env/` is layered (see ADR 0003):

- `resources_server/core.py`, `sandbox/`, `rewards/` — **zero `nemo_gym` imports**, run anywhere
- `resources_server/app.py`, `agent/app.py` — thin NeMo Gym adapters, imported only when the
  `gym` extra is installed

`uv` is also configured with `override-dependencies = ["uvloop; sys_platform != 'win32'"]`, which
lets the `gym` extra resolve on Windows **without editing one line of vendored NeMo Gym source**
(§1.3, §9.5). `uvloop` is never imported by NeMo Gym — it is a declared dependency only.

## Consequences

- ✅ `uv sync` completes in seconds on any OS with any Python ≥3.12; `make validate`,
  `make verify-gold`, `make test` and `make smoke` all work with no extra install.
- ✅ The verifier that produces the RL reward is the *same object* the eval path calls, because it
  lives in the core layer and both paths import it (§5.6 critical invariant).
- ⚠️ `make smoke-gym` (the real multi-process HTTP path) needs `uv sync --extra gym` and a
  POSIX-ish environment. Documented in `README.md` and `docs/nemo_gym_notes.md`.
- ⚠️ We must re-check the adapter against NeMo Gym `main` before the training phase, since the
  project explicitly warns "expect evolving APIs". Tracked in `docs/TRAINING_READINESS.md` §8.
