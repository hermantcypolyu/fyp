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

---

## 2026-08-01 · B3 batch 1 · The six agent tools (§5.5)

**Done:** `envs/coding_env/tools.py` — `list_files`, `read_file`, `write_file`, `edit_file`,
`run_tests`, `submit`, and nothing else. A `Toolbox` binds them to one materialised sandbox and one
task's `constraints:` block; `Toolbox.for_task(task, root)` is the factory. Write permission is a
pure string match against `benchmark.tools.globs`, in the same `repo/…`-rooted namespace `task.yaml`
already uses, so there is no path arithmetic at the decision point.

**Three properties the tests pin down:**
- **Refusals, not tracebacks.** Nothing raises out of a tool. Every failure is one sentence naming
  the file and the fix (`No match for old_str in repo/pkg/mod.py. Read the file and copy the text
  exactly.`), and `test_every_error_fits_the_short_budget` asserts that literally — a 1.5B model
  cannot act on a stack trace.
- **Deny beats allow.** `repo/pkg/conftest.py` matches both lists; it is refused, and it raises
  `tests_modified`. If allow had won, the §5.7 gate would be bypassable by picking the right path.
- **Two failures are gates, not errors.** Writing to `forbidden_write_globs` raises
  `tests_modified`; writing outside the sandbox root (checked *after* `resolve()`, so a symlink is
  caught too) raises `sandbox_escape`. Writing to an out-of-scope-but-harmless path — `repo/README.md`
  — is refused with **no** gate. Being out of scope is not cheating, and conflating the two would
  zero rewards for honest mistakes.

**Truncation is unconditional.** Every output goes through `truncate(text, budget)`: head 60% / tail
40% with an explicit marker, because the two informative parts of a pytest run are the first failure
and the final count. `read_file` additionally caps at 200 lines per call and returns
`continue with start=N`.

**`run_tests` cannot leak the grading signal**, for two independent reasons: the hidden suite is
never written into the agent's sandbox (`materialise(..., include_tests=False)`), and the tool
*refuses to run at all* if a `tests/` directory is nevertheless present. The second is the useful
one — it turns a sandbox-construction bug into a loud failure instead of a silently inflated score.

**Grounded in (spec §1.3), all @ `ec5a8136046aa0dd0c1525313654502f2fb7c42e`:**
- `resources_servers/legal_agent_bench/vendor/harvey_labs/lab_harbor/container_tool_runner.py` —
  `_ensure_allowed` (root containment), `_read`/`_slice_read_content` (paging + line cap),
  `_write`, `_edit` (exact match, count-aware errors)
- `resources_servers/legal_agent_bench/vendor/harvey_labs/lab_harbor/tools.py` — `TOOL_DEFINITIONS`
  JSON schema shape
- `resources_servers/browsecomp_advanced_harness/app.py` — `_bash_truncate` head/tail truncation
- `resources_servers/example_multi_step/app.py` — tool routing off a `SimpleResourcesServer`

**Commands run:** `uv run pytest -q` · `uv run ruff check .` · `uv run ruff format envs/`

**Numbers (measured):** **81 tests passed** (32 new in `envs/tests/test_tools.py`); ruff **0
findings**. No task, sandbox or benchmark file was touched; held-out untouched.

**Next:** B3 batch 2 — the sandbox and its isolation proof: agent copy built with
`include_tests=False`, verifier restoring pristine tests from source before grading, no network,
CPU/memory caps, hard timeout (timeout = failure, reward 0), and a test proving the hidden suite is
both unreadable and unwritable from inside the sandbox.

---

## 2026-08-02 · B3 batch 2 · Sandbox, verifier, isolation proof [security critical]

**Done:** `envs/coding_env/{sandbox,verifier,limits}.py` and 36 new tests.

**Isolation is absence, not permissions.** `Sandbox.start` materialises with
`include_tests=False`, so the hidden suite is never written into the agent's world. That is
stronger than a permission check, because it cannot be defeated by a hole in the check — there is
no file to reach. Three layers, in the order an attacker meets them: **absence** (nothing on disk),
**containment** (`Toolbox._resolve` rejects absolute paths and, *after* `resolve()`, anything
outside the root), **refusal** (`run_tests` will not run if a `tests/` directory appears anyway).

**The verifier never trusts the sandbox.** `_build_grading_copy` materialises a *fresh* copy from
`benchmark/tasks/…` — so the suite that runs is byte-identical to git — and only then copies the
agent's `repo/` over it. `test_verifier_ignores_tests_planted_in_the_sandbox` plants a suite that
passes trivially and asserts the score is unchanged: still 12 collected, still < 1.0.

**Gates (5.7), all zeroing rather than penalising:** `tests_modified`, `no_submit`,
`sandbox_escape`, `syntax_broken`, plus `timeout`. `pass_fraction` is kept alongside the zeroed
`reward` so the *cost* of each gate is measurable rather than invisible. A timeout is a scored
failure that still emits a row — raising instead would drop the slowest rollouts out of the dataset
and bias every aggregate computed from it. `syntax_broken` is `ast.parse`, not import: importing
runs module-level code, which for a broken repo can hang or write files.

**One hole found by writing the test, not by reading the code:** the verifier also gates on a
*planted pytest hook*. The write globs already forbid `**/conftest.py`, but a `conftest.py` written
straight to disk would otherwise have let `pytest_collection_modifyitems` empty the suite and score
1.0 on zero tests. `_has_injected_hook` diffs against the pristine repo, so a task that legitimately
ships one keeps it.

**Honest limits.** `ResourceLimits.enforced()` reports per-cap truth for the running platform and
the verifier records that dict on **every** result. On this Windows box only the wall clock and the
network guard actually hold; `cpu_s`, `memory_mb`, `max_file_mb` and `max_processes` are POSIX
`setrlimit` and are reported as **False** here. Claiming otherwise in the write-up would be exactly
the fabrication §1.2 forbids.

**Bug found by running it:** the first network guard replaced `socket.socket` outright. `ssl.py`
does `class SSLSocket(socket)` at import, so every graded run died with an INTERNALERROR from
`pdb → asyncio → ssl` before collecting a single test — reward 0 for correct solutions. The guard
now patches `connect`/`connect_ex`/`create_connection`/`getaddrinfo` and leaves the class alone;
`test_network_guard_survives_importing_ssl` is the regression.

**Symlink vector is actually proven, not skipped.** `os.symlink` needs Developer Mode on Windows, so
the fixture falls back to a directory junction (`mklink /J`, no elevation). Without that fallback the
one vector most likely to defeat a naive path check would have silently skipped on the machine this
is developed on.

**Single runner preserved.** `task_loader.run_tests` gained three optional, default-off hooks
(`extra_pythonpath`, `env_overrides`, `preexec_fn`) rather than the verifier growing its own copy of
the pytest invocation — 5.6's invariant is that eval and RL reward come from the same runner, and a
second copy would be free to drift.

**Commands run:** `uv run pytest` · `uv run ruff check .` · `uv run ruff format .` ·
`uv run python scripts/verify_gold_patches.py --task bugfix_metric_zero_domain_error`

**Numbers (measured):** **117 tests passed** (36 new: 22 isolation, 14 verifier/limits), 0 skipped;
ruff **0 findings**. One-task gold check still passes (6/12 untouched → 12/12 with gold, floor 0.50),
confirming the `run_tests` signature change did not disturb the authoring path. Corpus untouched at
**26 train / 0 held-out**; the hostile-episode test asserts the task directory is byte-identical
after every attack vector fires.

**Known limits, recorded not hidden:**
- The network guard is a `sitecustomize.py` patch, not a firewall. A subprocess started with a
  different interpreter, or a C extension calling `connect(2)`, walks past it.
- CPU/memory/fsize/nproc caps do not exist on Windows. They will engage unchanged on a Linux
  training box; until then, `limits_enforced` says so on every row.
- Hardlinks would defeat the `resolve()` containment check, but no tool creates one and the surface
  is exactly six tools.

**Next:** B3 batch 3 — resources server (`POST /verify`) and the agent harness as a thin subclass of
NeMo Gym's `simple_agent`.

---

## 2026-08-02 · B3 batch 3 · Resources server, agent harness, and two follow-ups

### Follow-ups from batch 2

**1. `test_count_mismatch` is now a hard gate.** If the graded run collects a different number of
tests than `task.yaml` declares, the episode is invalid: reward 0, recorded. `reward_v1` is a
*fraction*, so shrinking the denominator pays exactly as well as passing tests — a skip mark, a
collection hook, an early `sys.exit`, a deleted parametrize case. Pinning the denominator to the
number in git makes all of them worth zero without enumerating them, which is strictly better than
`_has_injected_hook` catching the one instance it recognises. It also catches a stale `n_tests`,
which is the same failure wearing a different hat. Verified the gate cannot misfire:
`validate_tasks.py --all` reports **26 tasks, 0 errors**, and it has an `N_TESTS` check that
compares the declared count against a real pytest run.

**2. ADR 0006 now states the coupling as binding.** Normalisation and moving the gate constant to
−1.0 must land in **one commit**. Shipping the formula alone creates a window where writing to
`tests/**` and doing nothing both score `0.0` — the deterrent for the #1 reward hack silently
disappears, nothing errors, no test goes red, and every run made in that window is unusable. Two
mechanical safeguards are specified with it: the gate constant derives from the same scale
definition as the formula, and a unit test asserts `gate_reward < normalised_reward(p=c)`.

### Batch 3 proper

**Where the logic lives.** `nemo_gym/server_utils.py` has a top-level `import resource`, so
`app.py` needs POSIX or B0's out-of-tree `_vendor/win_compat` shim — neither of which `uv sync`
provides. So `envs/coding_env/session.py` holds *all* the server's behaviour, framework-free and
covered by 21 tests, and `app.py` is a routing table over it. `envs/tests/test_server_contract.py`
then checks `app.py` and `agent.py` with `ast` rather than importing them: six routes named exactly
as the tool schema says, no second scoring path, no loop in the agent subclass.

**But the server was not left as a claim.** `scripts/smoke_one_task.py --transport gym` builds the
real FastAPI app and drives `/seed_session → six tool routes → /verify` through Starlette's
TestClient, using the vendored Python 3.12.10 and the shim. **reward 1.00, 12/12, 0 gates, 5 tool
calls, 0 live sessions afterwards** — byte-identical to the same episode via `--transport direct`.
That is §5.6's single-verifier invariant demonstrated rather than asserted: both transports reach
`EpisodeSession.grade()`, which calls `verifier.verify()` once and caches, so eval and reward
cannot disagree even in principle.

**The agent is genuinely thin.** `CodingAgent` overrides exactly one method and contains no loop —
a test asserts both. Its only job is the thing the built-in cannot do: `simple_agent` reads its step
cap from process-wide config, while our corpus varies `max_turns` per task (**20 / 25 / 30**, and
`max_tool_calls` **40 / 50 / 60** — measured, not assumed). It reads the per-rollout cap from the
row's `metadata` and hands the built-in loop a *config copy* carrying it; a copy rather than a
mutation, because one server object serves concurrent rollouts and assignment would leak one task's
budget into another's.

**Budgets are enforced in different places, deliberately.** `max_tool_calls` is server-side, because
an agent must not be able to opt out of an environment limit and only the resources server sees
every call. `max_turns` is agent-side, because a turn is a loop iteration and only the loop can
count them.

**`make smoke` now exists.** The Makefile has referenced `scripts/smoke_one_task.py` since B0; it is
written. The "agent" is scripted — it writes the gold solution *through the real `write_file` tool*
— because a model would test the model, whereas this tests the environment.

**One correction to batch 2's write-up:** I claimed `nemo_gym` "cannot be imported on Windows at
all". With B0's shim it can, and now demonstrably does. The docstrings that said otherwise are
fixed.

**Commands run:** `uv run pytest` · `uv run ruff check .` · `validate_tasks.py --all` ·
`smoke_one_task.py --task bugfix_metric_zero_domain_error` (both transports)

**Numbers (measured):** **147 passed, 1 skipped** (48 new); ruff **0 findings**; validator **26
tasks, 0 errors, 5 warnings** (the known mix/provenance gaps). Smoke: **1.00 / 12 of 12** on both
transports. `limits_enforced` on the HTTP result reports `wall_clock_s` and `network` true,
the four `setrlimit` caps false — correct for Windows.

**Artifacts:** `results/raw/b3_smoke/{direct,gym}.json`

**Next:** B3 batch 4 — trace persistence per `analysis/trace_schema.json`, pinned SHAs for
`_vendor/nemo-gym` and NeMo RL, and `docs/SETUP_NEW_MACHINE.md`.

---

## 2026-08-02 · B3 batch 4 · Trace + portability — **B3 CLOSED**

**Traces.** `analysis/trace_schema.json` (draft 2020-12) plus `analysis/trace.py`. Every field is
populated by a real run — the schema was written first, then the smoke test was made to satisfy it,
not the other way round. Two decisions:

- **The row is self-contained.** Category, provenance, budgets, per-test outcomes, the commits of
  every moving part and which resource caps the host actually applied are copied in rather than
  looked up later. A trace that needs `benchmark/tasks/` at its original revision stops being
  interpretable the moment a task is edited.
- **Validation happens on write.** A malformed row found during analysis is a run to repeat; found
  at write time it is a bug fixed in seconds. `write_jsonl` re-validates, so a hand-built row cannot
  slip in either.

Tests target the *rejections* — missing field, unknown gate name, stray key, out-of-range
`pass_fraction` — because a silently accepted bad row looks fine until someone tries to use it.
`test_a_real_episode_produces_a_valid_row` runs an actual episode rather than validating a row
written to match the schema it is checked against.

**Pinned versions.** `analysis/versions.py` + `docs/pinned_versions.md`:

| Stack | Commit | `requires-python` |
| --- | --- | --- |
| NeMo Gym | `ec5a8136046aa0dd0c1525313654502f2fb7c42e` (2026-07-31) | `>=3.12` |
| NeMo RL | `55296257c96d49cd95c7d77613cb0f36bd4a4dc7` (2026-08-02) | `>=3.13.13,<3.14` |

NeMo RL cloned to `_vendor/nemo-rl` (23 MB, `--depth 1`), **not installed, not run, not imported**.

⚠️ **Finding: the two stacks do not share a Python version.** NeMo RL requires **3.13.13 ≤ py <
3.14** and will not install on 3.12, while NeMo Gym needs 3.12 here (`ray` publishes no
`win_amd64` wheel past cp312). A training box needs **two interpreters**. This contradicts the
"Python 3.12 for everything" assumption in the batch brief; `SETUP_NEW_MACHINE.md` §1 and §3 say so
explicitly, and a unit test pins both strings so the fact cannot quietly disappear.

**Drift is detected, not assumed.** `versions.resolve()` reads the live `HEAD` of each vendored
clone and every trace records `{pinned, actual, matches}`. Recording only the pin lets a clone
drift silently; recording only the actual commit makes an old result uninterpretable. A test asserts
both docs quote the same SHAs as the code, because the clone commands are what a human copy-pastes.

**`docs/SETUP_NEW_MACHINE.md`** — cold start in five steps, then §6, marked **REQUIRED**: re-check
`ResourceLimits.enforced()` on Linux. Four of six caps are POSIX `setrlimit`, `preexec_fn()` returns
`None` off POSIX, so `cpu_s`, `memory_mb`, `max_file_mb` and `max_processes` have **never executed
even once**. The doc gives the expected output (all six `true`) and four probes that make each cap
actually bite, because "the call did not raise" is not "the limit applied". Until that is done they
are *claimed, not enforced* — which is exactly why `limits_enforced` rides on every trace row
instead of sitting in a config nobody reads. §7's hardware table separates **measured** (CPU-only
work, ~40 s) from ***projected*** (every GPU figure — there is no NVIDIA GPU on this machine).

**Also this batch:** `/verify` now returns `per_test`, so the HTTP transport can build a complete
trace from what the wire actually returned rather than from a second local re-run.

**Commands run:** `uv run pytest` · `uv run ruff check .` · `smoke_one_task.py --trace` on both
transports · `git check-ignore -v`

**Numbers (measured):** **168 passed, 1 skipped** (21 new); ruff **0 findings**. Smoke **1.00 /
12 of 12** on both transports, each emitting a schema-valid trace. Ignore rules confirmed:
`.venv/` → `.gitignore:23`, `_vendor/` → `:72`, `results/**/*.jsonl` → `:54`.

**Artifacts:** `results/raw/b3_smoke/{direct,gym}.json`, `trace_example.json` (one full row,
committed because the `.jsonl` is correctly gitignored).

---

### B3 exit criteria (spec §4)

| Criterion | State |
| --- | --- |
| Resources server exposing the §5.5 tool surface + `POST /verify` | ✅ driven end-to-end over HTTP, reward 1.00 |
| Agent harness as a thin subclass of a built-in | ✅ `CodingAgent`: one method, no loop, both asserted |
| Sandbox: copy, hidden tests outside writable root, pristine restore, no network, caps, timeout | ✅ — caps are POSIX-only and **unexercised**, see `SETUP_NEW_MACHINE.md` §6 |
| Rollout trace persisted exactly per `analysis/trace_schema.json` | ✅ validated on write, both transports |
| `make smoke` runs one task end-to-end and returns a real reward | ✅ `PASS … reward=1.00 tests=12/12` |
| Isolation test proves the agent cannot read or write hidden tests ← hard gate | ✅ 22 tests, five vectors, 0 skipped |

**Carried into B4, not silently dropped:**
1. Resource caps unverified on POSIX (above).
2. Ray/`gym env start` unverified — Windows cannot boot the cluster.
3. Corpus is **26 train / 0 held-out**, below §5.2's 30 + 10, and **42% `real_repo`** against a
   ≥50% target. B2 was closed by decision, not by hitting the target.
4. ADR 0006 remains **Proposed**; normalisation and the −1.0 gate constant must land together.

**Next:** B4 — `rewards/components.py` and `compose.py` (reward_v1 + the gates as pure, unit-tested
functions), `analysis/metrics.py` (pass@k + bootstrap CI), `analysis/failure_taxonomy.py`, and
`docs/reward_design.md`.

---

## 2026-08-02 · Held-out batch 1 · 3 of 10 tasks — **NOT SEALED**

| task | cat | diff | untouched | +gold | floor | headroom |
| --- | --- | --- | --- | --- | --- | --- |
| `bugfix_image_alt_double_escape` | bug_fix | easy | 5/13 | 13/13 | 0.38 | 8 |
| `bugfix_retry_base_composition` | bug_fix | medium | 6/14 | 14/14 | 0.43 | 8 |
| `testupdate_rpartial_rcurry_rename` | test_update | medium | 6/13 | 13/13 | 0.46 | 7 |

All three `source: real_repo`, from upstream commits, extracted in the same direction as the train
corpus (buggy file from the commit's **parent** for bug_fix; correct library from the **commit** and
stale suite from the **parent** for test_update).

**`verification.floor_tests` added to the schema — measurement only.** It records how many hidden
tests already pass against the untouched repo. `validate_tasks.py` checks it when present and
`--fix-counts` writes it, so it is a *checked measurement* rather than a number an author typed.
`reward_v1` is unchanged and ADR 0006 remains **Proposed** — nothing normalises against this yet.
It is optional, so the 26 train tasks are unaffected; they get backfilled in B4.

Every task clears the ≥6 headroom bar (8 / 8 / 7). Floors sit at 0.38–0.46 against the train mean of
0.46, so the held-out split is not systematically easier or harder — which is the point of §5.4
rule 7.

**One subagent claim I checked and had to correct.** The `test_update` author reported that
`thefuzz` is GPL-2 and that `mine_commits.py` misreports it as MIT. `_vendor/mining/thefuzz/
LICENSE.txt` reads `MIT License / Copyright (c) 2014 SeatGeek`. The tool was right; the report was
wrong. Left unchecked, that would have removed a permissively-licensed repo from the pool for the
remaining seven tasks.

**Corpus after this batch:** 26 train + 3 held-out = **29 tasks, 0 errors**, all solvable and
failable. `real_repo` up from 42% to **48%** (14/29). Held-out still needs: bug_fix 2, test_update 1,
config_fix 1, small_feature 2, docs_improve 1 — and the difficulty spread needs the remaining slots
to lean easy/hard, since the split is currently 1 easy / 2 medium / 0 hard against a 40/40/20 target.

**Commands run:** `validate_tasks.py --all` · `verify_gold_patches.py --split heldout` ·
`uv run pytest` · `uv run ruff check .`

**Numbers (measured):** validator **29 tasks, 0 errors, 12 warnings** (all corpus-mix/provenance,
expected at 3/10 held-out); gold patches **3/3 solvable and failable**; unit tests **168 passed,
1 skipped**; ruff **0 findings**.

**Artifacts:** `results/raw/b2_heldout/batch1.json`

**Held-out is NOT sealed.** No commit SHA recorded here; the split stays open until all 10 are
reviewed.
