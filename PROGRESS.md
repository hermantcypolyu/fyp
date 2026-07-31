# PROGRESS

Append-only. One block per session. Never edit past entries.

---

## 2026-07-31 · B0 · Scaffolding + NeMo Gym grounding

**Goal:** Repo skeleton that `uv sync`s from cold, real hardware facts on disk, and
`docs/nemo_gym_notes.md` written from the actual NeMo Gym source rather than memory.

**Done:**
- Repo scaffolded per §3. `uv sync --extra dev` succeeds from clean (19 packages). Core deps
  deliberately tiny — `pyyaml`, `jsonschema`, `numpy` — with `nemo-gym` behind an optional extra
  (ADR 0002).
- `Makefile` + `make.ps1` (Windows shim) with `install / validate / verify-gold / catalogue /
  smoke / smoke-gym / test / lint / fmt / check / freshclone / clean`.
- NeMo Gym cloned at `ec5a8136046aa0dd0c1525313654502f2fb7c42e` (v0.5.0rc0), read in depth.
  **Two built-in environments run unmodified** end to end through their real FastAPI apps via
  `scripts/check_nemo_gym_install.py`. Their own pytest suites also pass unmodified.
- `docs/nemo_gym_notes.md` written from source with file:line citations — server registration,
  tool declaration, `/verify` contract, agent loop, dataset shape, config wiring, plus a survey of
  the hub (101 resources servers / 33 agents) and which harness we build on.
- `docs/vram_notes.md` written honestly: **this build machine has no NVIDIA GPU.** All 4070 Ti
  figures marked `[NOT MEASURED]`; `scripts/capture_hardware.py` added so the real numbers are one
  command away.
- ADRs 0001–0005 written.

**Commands:**
- `uv sync --extra dev`
- `uv run python scripts/capture_hardware.py --out results/raw/b0_hardware/build_machine.md`
- `python scripts/check_nemo_gym_install.py --out results/raw/b0_nemo_gym_builtin/report.json`
- `pytest resources_servers/example_single_tool_call/tests/test_app.py` (and `example_multi_step`,
  `example_session_state_mgmt`) inside the vendored checkout

**Artifacts:**
- `results/raw/b0_hardware/build_machine.md`
- `results/raw/b0_nemo_gym_builtin/report.json`

**Numbers** (all reproducible from `results/`):
- `example_single_tool_call`: tool 200, `/verify` → **reward 1.0**
- `example_session_state_mgmt`: seed(10) → +5 → +7 → counter **22**, `/verify` → **reward 1.0**,
  1 live session
- Built-in test suites: **3 files, 1 passed each, 0 failed**
- Build machine: Windows 11, 12 logical cores, 31.4 GiB RAM, 790 GiB free, **0 NVIDIA GPUs**
- NeMo Gym hub: **101 resources servers, 33 agents**

**Blocked / needs human:**
1. **No NVIDIA GPU on this machine.** `docs/vram_notes.md` §2 and every VRAM figure stay
   `[NOT MEASURED]` until someone runs `scripts/capture_hardware.py` on the RTX 4070 Ti.
2. **`gym env start` cannot run here.** Ray's raylet times out on native Windows
   (`GCS cannot find the node ...`, `nemo_gym/server_utils.py:492`); NeMo Gym supports Windows via
   WSL2 only. So the multi-process path (`gym eval run`, real sockets, process supervision) is
   **unverified**. Everything we subclass *is* verified via the in-process route. Clearing this
   needs one `make smoke-gym` run on Linux/WSL2 — see `docs/TRAINING_READINESS.md` §5.
3. Windows needed three workarounds, all confined to `../_vendor/` with **zero edits to NeMo Gym
   source**: a POSIX `resource` shim, a `uvloop` marker override, and Python 3.12 for `ray`'s
   wheel coverage. Details in `docs/nemo_gym_notes.md` §7.

**Next:** B1 — `benchmark/schema/task.schema.json` per §5.1, `validate_tasks.py` with actionable
errors, `benchmark/tools/new_task.py`, and 3 fully-worked reference tasks (`bug_fix`,
`test_update`, `config_fix`) each with a passing gold patch and a failing untouched repo.
