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


## 2026-07-31 · B1 · Task schema, authoring tooling, 3 reference tasks

**Goal:** A task format that cannot be got wrong quietly — a schema, a validator whose error
messages tell you the fix, a scaffolder, and three fully-worked reference tasks that prove the
format carries the three most different categories.

**Done:**
- `benchmark/schema/task.schema.json` — Draft 2020-12, `additionalProperties: false` throughout,
  every §5.1 field constrained (id pattern, `repo.loc` ≤ 500, `n_tests` 3–40, conditional
  `origin`+`notes` when `provenance.source: real_repo`).
- `benchmark/schema/validate_tasks.py` — 10 check families, 34 distinct error codes, each with a
  remediation line. Crucially it *measures* rather than trusts: it counts `repo/` LOC, collects the
  hidden suite, runs it twice, and cross-checks `n_tests`, `expected_files_changed` and the write
  globs against what the gold patch actually touches. It caught all three bugs in the first
  reference task before a human read it.
- `benchmark/tools/task_loader.py` — the single place that materialises a disposable working copy
  and runs pytest against it. `include_tests=False` is how B3's sandbox will be built: the hidden
  suite is simply never written into the agent's world, so isolation does not depend on file
  permissions. YAML dates are coerced to ISO strings on load so authors need not remember quotes.
- `benchmark/tools/patch_utils.py` — pure-Python unified diff (parse / apply / reverse / dry-run /
  generate). We own this rather than shelling out to `git apply` so behaviour is identical on a
  Windows laptop with no git and on a Linux training node; it also normalises CRLF, which `git
  apply` does not.
- `benchmark/tools/globs.py` — glob matcher with explicit `**` semantics. `fnmatch` treats `/` as an
  ordinary character, which would silently widen every task's write permissions; since these globs
  become B4's `tests_modified` hard-zero gate, the semantics are spelled out and tested.
- `benchmark/tools/new_task.py` (+ `make new-task`) — scaffolds a task with per-category defaults
  and the mandatory deny-list pre-filled. Verified: the generated stub *fails* validation on 5
  counts, so a half-finished task cannot enter the corpus.
- `benchmark/tools/make_gold_patch.py` (+ `make gold-patches`) — generates `solution/gold.patch` by
  diffing `solution/fixed/` against `repo/`. Gold patches are never hand-written.
- `scripts/verify_gold_patches.py` (+ `make verify-gold`) — runs each hidden suite untouched and
  gold-patched, and reports the **reward floor**: what an agent scores for doing nothing.
- Three reference tasks, all green:
  | id | category | difficulty | untouched | + gold | floor |
  | --- | --- | --- | --- | --- | --- |
  | `bugfix_offbyone_pagination` | bug_fix | easy | 6/11 | 11/11 | 0.55 |
  | `testupdate_money_api_v2` | test_update | medium | 4/10 | 10/10 | 0.40 |
  | `configfix_textkit_packaging` | config_fix | hard | 0/9 | 9/9 | 0.00 |
- 37 unit tests for the patch engine and the glob matcher, including a regression test for the
  parser bug below and a test asserting that the hidden suite at `tests/**` and a project's own
  visible suite at `repo/tests/**` never collide — the invariant every `test_update` task rests on.
- `docs/task_authoring_guide.md` — complete enough to author task #4 without reading any tooling.

**Commands:**
- `uv run python benchmark/tools/new_task.py <id> --category <c>`
- `uv run python benchmark/tools/make_gold_patch.py --all`
- `uv run python benchmark/schema/validate_tasks.py --all`
- `uv run python scripts/verify_gold_patches.py --split all --json results/raw/b1_gold_patches/report.json`
- `uv run pytest -q`  ·  `uv run ruff check .`  ·  `uv run ruff format .`

**Artifacts:**
- `results/raw/b1_gold_patches/report.json`
- `benchmark/tasks/train/{bugfix_offbyone_pagination,testupdate_money_api_v2,configfix_textkit_packaging}/`

**Numbers (all measured, nothing estimated):**
- Validator: **3 tasks, 0 errors, 7 warnings** (all 7 are corpus-mix warnings — expected at 3/30)
- Gold patches: **3/3 solvable and failable**; reward floor min 0.00 / mean 0.32 / max 0.55
- Unit tests: **37 passed**; ruff: **0 findings**
- Corpus so far: 3/30 train, 0/10 held-out; 0% `real_repo` (target ≥50%)

**Blocked / needs human:** nothing new. B0's two open items stand (no NVIDIA GPU here; `gym env
start` unverified because Ray cannot start on native Windows).

**Bugs found and fixed during B1** — all three surfaced by running the pipeline, none by reading:
1. `parse_unified_diff` absorbed the trailing empty string from `text.split("\n")` as a phantom
   blank context line, so any patch whose last hunk ran to end-of-file failed with a bogus
   "context not found". Hunks now stop at the line counts declared in the `@@` header.
2. Unquoted `created: 2026-07-31` parsed as a `datetime.date` and failed the schema's `string`
   type. Fixed in the loader rather than in every task file.
3. `globs.normalise` did not collapse `//` or a trailing `/`, so `tests//test_x.py` would have
   stepped straight around the `tests/**` deny-list. Found by a unit test, not by the pipeline —
   which is the argument for having written the unit tests.

**Next:** B2 — grow the corpus to 30 train + 10 held-out at the §5.2 mix, in batches of 5–8, with
**≥50% `source: real_repo`** mined from permissively-licensed Python projects (`git clone` works on
this network even though release-binary downloads do not). Then `build_catalogue.py`,
`benchmark/catalogue.md`, the JSONL build artefacts, and sealing the held-out split by recording
its commit SHA here.
