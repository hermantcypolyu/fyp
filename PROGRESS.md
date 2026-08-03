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

---

## 2026-08-03 · Held-out batches 2–4 · measurement + two corrections — **NOT SEALED**

Floors measured for the remaining seven tasks via `validate_tasks.py --fix-counts`. They do **not**
meet the ≥6 headroom bar that batch 1 cleared:

| task | cat | diff | n | floor | headroom |
| --- | --- | --- | --- | --- | --- |
| `bugfix_moreitertools_window` | bug_fix | hard | 3 | 2 | 1 |
| `configfix_pkg_cli_entrypoint` | config_fix | easy | 3 | 2 | 1 |
| `feature_inflect_plural_forms` | small_feature | medium | 3 | 1 | 2 |
| `feature_thefuzz_normalize` | small_feature | easy | 3 | 1 | 2 |
| `bugfix_slugify_empty_string` | bug_fix | easy | 4 | 1 | 3 |
| `docs_readme_cli_examples` | docs_improve | hard | 3 | 0 | 3 |
| `testupdate_thefuzz_api_rename` | test_update | medium | 3 | 0 | 3 |

**Limitation — held-out/train repo overlap.** Two held-out tasks reuse a library the train split
already draws from, so they are not fully independent measurements:

- `tenacity` — train `bugfix_ordinal_suffix_teens` ↔ held-out `bugfix_retry_base_composition`
- `funcy` — train `bugfix_throttle_timedelta`, `bugfix_where_missing_keys`,
  `testupdate_funcy_omit_rename` ↔ held-out `testupdate_rpartial_rcurry_rename`

A further three (`bugfix_moreitertools_window`, `feature_thefuzz_normalize`,
`testupdate_thefuzz_api_rename`) carry `more-itertools`/`thefuzz` in their *names* only — they are
authored stubs sharing no code with train's `testupdate_always_reversible_rename`. The name overlap
is cosmetic; the code overlap is `tenacity` and `funcy`. Only `mistune` is genuinely unseen.

**Correction — provenance was overstated.** `bugfix_slugify_empty_string` and
`feature_thefuzz_normalize` were marked `source: real_repo` with upstream origins in the previous
session. Their repos are 3–8 line authored stubs, not code derived from those projects. Both are
back to `source: authored` with the origin removed. `real_repo` share is therefore **3/10 (30%)**
held-out, not 5/10 — below the §5.3 ≥50% target, and the validator now warns accordingly.

**Commands run:** `validate_tasks.py --split heldout --fix-counts` ·
`verify_gold_patches.py --split heldout`

**Numbers (measured):** validator **10 tasks, 0 errors, 1 warning** (provenance 30%); gold patches
**10/10 solvable and failable**; reward floor min 0.00 / mean 0.35 / max 0.67.

**Held-out is NOT sealed.** ADR 0006 not implemented; no training run.

---

## 2026-08-03 · Held-out batch 2 · 3 stubs replaced with real upstream tasks — **NOT SEALED**

The padding option was rejected: assertions that co-flip on one edit raise `n_tests - floor_tests`
on paper while reward stays binary, which would have hidden the failure being measured. These three
are re-authored from real commits instead, extracted in the same direction as batch 1.

| task | cat | diff | LOC | untouched | +gold | headroom |
| --- | --- | --- | --- | --- | --- | --- |
| `bugfix_mistune_angle_link_escape` | bug_fix | hard | 255 | 9/17 | 17/17 | 8 |
| `bugfix_mistune_codespan_backticks` | bug_fix | medium | 264 | 6/13 | 13/13 | 7 |
| `testupdate_thefuzz_ascii_only_rename` | test_update | medium | 136 | 10/16 | 16/16 | 6 |

Headroom is honest in each: the mistune link bug is independently reachable through inline links,
reference definitions, an escaped `<`/`>` and a dangling backslash; the codespan bug splits into
fence *sizing* and space *padding*, so half a fix scores strictly between floor and 1.0; the thefuzz
rename grades suite-passes, suite-not-gutted, new name used, each removed name gone, and the Python 2
shim removed — all separately achievable.

Stubs deleted: `bugfix_moreitertools_window`, `bugfix_slugify_empty_string`,
`testupdate_thefuzz_api_rename`.

**Mining findings — two categories cannot be filled from real upstream.**

- `jaraco/inflect` has **no licence file** (`mine_commits.py` reports "DO NOT USE"), so it cannot
  supply a `real_repo` task at all.
- `python-slugify` yields **zero** candidates in every kind; `itsdangerous` yields none for `bug_fix`.
- **`config_fix` and `docs_improve` return zero candidates** across all four licence-clean repos
  unseen by train (slugify, itsdangerous, thefuzz, mistune). The one near miss,
  mistune `2f2449dd0ab1` (adds `[project.scripts]` to `pyproject.toml`), is not extractable: its
  `__main__.py` imports the full renderer stack, so the slice would be an order of magnitude over
  the size budget.

So the remaining `config_fix` and `docs_improve` slots will stay `source: authored` and be rebuilt at
real size with real headroom, rather than being relabelled. Recorded here rather than forced.

**Provenance now 6/10 (60%) `real_repo`** — above the §5.3 ≥50% target, and the corpus-mix warning is
gone. No *new* train overlap: train uses neither mistune nor thefuzz. The tenacity/funcy overlap noted
above is unchanged. New concentration limitation: **mistune now supplies 3 of 10** held-out tasks.

Difficulty stands at 3 easy / 5 medium / 2 hard; one of the four remaining rebuilds must land easy
rather than medium to reach the 4/4/2 target.

**Commands run:** `validate_tasks.py --split heldout --fix-counts` ·
`verify_gold_patches.py --split heldout`

**Numbers (measured):** validator **10 tasks, 0 errors, 0 warnings**; gold patches **10/10 solvable
and failable**; reward floor min 0.00 / mean 0.42 / max 0.67.

**Held-out is NOT sealed.** ADR 0006 not implemented; no training run.

---

## 2026-08-03 · Held-out batch 3 · final 4 stubs rebuilt — 10/10 authored, **NOT SEALED**

| task | cat | diff | LOC | untouched | +gold | headroom |
| --- | --- | --- | --- | --- | --- | --- |
| `configfix_tallyman_command_paths` | config_fix | easy | 149 | 7/14 | 14/14 | 7 |
| `docs_interval_docstring_drift` | docs_improve | hard | 144 | 8/17 | 17/17 | 9 |
| `feature_pluralise_rule_classes` | small_feature | easy | 124 | 9/18 | 18/18 | 9 |
| `feature_textnorm_options` | small_feature | easy | 134 | 7/16 | 16/16 | 9 |

All four are `source: authored`, and none were relabelled. mistune was avoided, so the split is not
single-repo dominant.

**Why all four are authored, not real_repo.** `config_fix` and `docs_improve` return **zero**
candidates in every licence-clean repo unseen by train, as recorded in batch 2. For the two
`small_feature` slots the difficulty mix forced **easy**, and no unseen repo yields an easy feature
inside the size budget: `jaraco/inflect` — the obvious source for pluralisation — **ships no licence
file at all**, `python-slugify` returns no candidates in any kind, and every `itsdangerous` feature
commit sits in the serializer/signer stack, which does not slice below ~400 LOC.

**Partial credit is real in each.** The config task has two independent defects — entry points naming
a module a refactor removed, and a `packages` list that would not ship the subpackage holding them.
The docs task drifts six executable examples plus three prose claims, one per function. Both feature
tasks add several independently implementable behaviours, and both carry a floor test that an
over-broad fix *breaks*: `roof`/`chief`/`belief`/`cliff` already pluralise correctly and regress if
`-ves` is added without the exception list.

**Final held-out corpus:** 4 easy / 4 medium / 2 hard — the 4/4/2 target. Categories: bug_fix 4,
test_update 2, small_feature 2, config_fix 1, docs_improve 1. Provenance **6/10 (60%) real_repo**,
above the §5.3 ≥50% target. Headroom is **6-9 on every task**, minimum 6. Reward floor mean 0.48.

**Two things worth recording honestly.**

1. Three stub directories deleted in batch 2 had partially **reappeared on disk as untracked
   remnants** (each missing `repo/` or `tests/`), which made the validator see 13 tasks and fail. They
   were absent from `HEAD`, so the batch 2 commits were correct; the leftovers were removed and
   `__pycache__` directories cleaned out of two task repos.
2. Three of the four rebuilt repos are **124-149 LOC**, just under the 150-250 band asked for. They
   are genuine multi-module packages rather than stubs and sit alongside the accepted 136-LOC
   `testupdate_thefuzz_ascii_only_rename`, so the shortfall was left rather than closed by adding
   filler modules purely to move the number.

**Commands run:** `validate_tasks.py --split heldout --fix-counts` ·
`verify_gold_patches.py --split heldout` · `make_gold_patch.py --all --check`

**Numbers (measured):** validator **10 tasks, 0 errors, 0 warnings**; gold patches **10/10 solvable
and failable**, none stale; reward floor min 0.38 / mean 0.48 / max 0.62.

**Held-out is NOT sealed.** No commit SHA recorded as a seal; ADR 0006 not implemented; no training
run.

---

## 2026-08-03 · Step 1 · `floor_tests` backfilled for all 26 train tasks — **NOT SEALED**

Every task in the corpus now carries a **measured** `verification.floor_tests`, written by
`validate_tasks.py --fix-counts` rather than typed by an author. Cross-checked against
`verify_gold_patches.py --split train`: all 26 recorded floors equal the untouched pass count.
Train floors run 0-8; reward floor min 0.00 / mean 0.46 / max 0.58.

**A validator bug found while doing it — warnings were suppressing the behaviour check.**
The gate in front of `check_behaviour` read:

```python
if not args.no_run and schema_ok and not report.for_task(task.id):
```

`report.for_task()` returns *every* finding, warnings included. So any task carrying a warning had
`check_behaviour` skipped entirely - and that is the function which proves a task is solvable by its
gold patch and failable without it, **and** which measures `floor_tests`. The validator then printed
`ok` for a task it had never actually run.

Both oversized `test_update` train tasks were in exactly that state, on a `LOC_BIG` **warning**:
their floors read `0` when the measured value is `5`, and their solvable/failable checks had not run
in any corpus pass. The same silently applied to `bugfix_mistune_codespan_backticks` while it was
over 300 LOC during batch 2 (it was verified separately, so its numbers stood).

Fixed by gating on errors only, extracted as `blocking_errors()` so the invariant is testable, with
`benchmark/tests/test_behaviour_gate.py` pinning it: a warning alone must not block, an error must,
and an error mixed with warnings must still block. Nothing else about the validator changed.

**Step 0 (trim the two oversized train repos) — not possible without weakening. Left as is.**
In both tasks the two largest files *are* the graded surface, measured per file:

| task | LOC | must shed | largest files |
| --- | --- | --- | --- |
| `testupdate_always_reversible_rename` | 346 | 47 | `more.py` 198, `tests/test_more.py` 129, `__init__.py` 19 |
| `testupdate_funcy_omit_rename` | 376 | 77 | `colls.py` 130, `tests/test_colls.py` 112, then 5 files totalling 117 |

`more.py` holds `always_reversible` — the renamed symbol the task is about — plus five helpers
reachable from the visible suite; `tests/test_more.py` is the file the agent edits. That leaves only
`__init__.py` (19 LOC) as non-graded, against a 47 LOC shortfall. For funcy, deleting `funcs.py`,
`seqs.py` and `primitives.py` outright saves 62 LOC and still lands at 314 — and would break the
`__init__` exports and the `EMPTY` sentinel `colls.py` depends on.

This is the opposite of the `bugfix_mistune_codespan_backticks` case, where `Parser`/`InlineState`
were ~100 LOC of parser machinery inside a *renderer* slice that used only `state.env`. Here there is
no comparable dead weight, so the two `LOC_BIG` warnings are accepted as a stated limitation.

**Commands run:** `validate_tasks.py --all --fix-counts` · `verify_gold_patches.py --split train` ·
`uv run pytest`

**Numbers (measured):** validator **36 tasks, 0 errors, 5 warnings**; gold patches **26/26 train
solvable and failable**; unit tests **174 passed, 1 skipped**.

**Held-out is NOT sealed.** ADR 0006 not implemented; no training run.

---

## 2026-08-03 · Step 2 · ADR 0006 implemented — reward normalised, gates to −1.0 — **NOT SEALED**

ADR 0006 moved to **Accepted**, and all three of its coupled points landed in **one commit**:
the formula, the widened range, and the gate constant.

```
reward_v1 = (passed - floor_tests) / (n_tests - floor_tests)    ∈ [-1, 1]
gates                                                           = -1.0
```

**Measured across all 36 tasks after the change:**

| | before | after |
| --- | --- | --- |
| do-nothing (`passed == floor_tests`) | 0.467 mean, 0.00–0.58 | **0.0 on every task** |
| gold (`passed == n_tests`) | 1.0 | **1.0 on every task** |
| gated rollout, even when fully solved | 0.0 | **−1.0** |

A partial fix now reads as a fraction of the *movable* work. On the ten held-out tasks
(headroom 6–9), winning one test scores **0.11–0.17**, winning half the movable tests scores
**0.43–0.50**, and breaking one test that passed untouched scores **−0.11 to −0.17**. Under the raw
scale those same rollouts were compressed into a narrow band above 0.38–0.62 and could never go
negative.

**Why the coupling was load-bearing.** Before, a gate returned 0.0, far below every task's floor,
so writing to `tests/**` was strictly worse than doing nothing. After normalisation `0.0` *is*
doing nothing — so shipping the formula alone would have made the #1 reward hack score exactly
what an immediate submit scores. Nothing would have errored and no test would have gone red.
Two mechanical safeguards, both required by the ADR:

- `rewards/compose.py` defines the scale once and **derives** `GATE_REWARD` from `REWARD_MIN`,
  so the constant cannot be left behind when the formula changes.
- `rewards/tests/test_compose.py::test_a_gated_rollout_scores_below_doing_nothing` asserts
  `GATE_REWARD < normalised_reward(p = floor_tests)` across five task shapes.

**Negative rewards are safe in the advantage computation.** No advantage code exists yet — GRPO's
$(r-\mu_g)/\sigma_g$ is specified in ADR 0006 and `docs/vram_notes.md` but unimplemented — so the
check is that nothing downstream assumes $r \ge 0$. Audited: the trace schema bounded `reward`
implicitly and now states `[-1, 1]` explicitly; `pass_fraction` keeps its `[0,1]` bound because it
is still the raw fraction; the `/verify` response model is unbounded. A unit test covers the
arithmetic directly — a mixed group `[0.0, 0.5, 1.0, −1.0]` has non-zero σ, produces no NaN, and
puts the gated rollout at the most negative advantage. Group normalisation is sign-agnostic; the
wider range strictly *increases* spread, which is the intended effect.

**Trace schema bumped `1` → `2`**: `reward` range widened and `result.floor_tests` added, so every
rollout's reward is reconstructible from the raw measurement. `read_jsonl` deliberately does not
validate, so the two existing v1 smoke traces stay readable.

**Transports.** `--transport direct` re-run and verified: reward **1.0**, `floor_tests` 7, v2 trace
written. `--transport gym` **could not be re-run on this machine** — `nemo-gym` depends on `uvloop`,
which has no Windows build (ADR 0002), so the extra does not install here. That path is verified the
way the rest of the gym surface already is, statically: a new contract test asserts
`CodingVerifyResponse` declares `floor_tests` **and** that `app.py` populates it from the result,
because a dropped field would silently make every gym-side reward the raw pass fraction again — the
exact regression this ADR exists to prevent, on one transport only. `results/raw/b3_smoke/
rollouts_gym.jsonl` remains a **v1 artefact from B3 and must be regenerated on Linux/macOS**; its
headline reward is unaffected, since a gold rollout scores 1.0 under both the old and new formulas.

**Commands run:** `uv run pytest` · `ruff check .` · `smoke_one_task.py --transport direct`

**Numbers (measured):** unit tests **191 passed, 1 skipped** (16 new reward tests); ruff **0
findings**; direct smoke **reward 1.0**.

**Held-out is NOT sealed.** No training run; no weights downloaded.

---

## 2026-08-03 · Step 3 · **HELD-OUT SEALED** · B4

| | |
| --- | --- |
| Sealed | 2026-08-03 |
| Held-out tree SHA | `318fedc8868bcdd46f24dd78793a5c136be54400` |
| Last commit to modify the split | `ad46f8746a1db72cc925dd656fdb1b7a5ed85d47` |
| Sealing commit | recorded in the follow-up commit — see "Why the anchor is a tree hash" below |
| Record | `benchmark/heldout_seal.json`, `docs/heldout_seal.md` |
| Enforcement | `scripts/check_heldout_seal.py`, wired into `make check` |

**From this point, for the rest of the project: no edits, no inspection, no use for
checkpoint selection, no use for hyperparameter tuning.** The split is not read, browsed or
sampled; per-task held-out results are not examined to form hypotheses; held-out reward never
chooses a checkpoint, triggers early stopping, or tunes a learning rate, group size, KL
coefficient, turn budget or prompt. It is measured once, at the end, and reported whatever it
says. **Checkpoint selection and all tuning use a validation slice carved from `train` only** —
the 26 train tasks are the entire model-selection budget.

**Why the anchor is a tree hash, not a commit SHA.** A commit cannot contain its own SHA, so
"the sealing commit" cannot be the thing the check compares against without a circular
reference. The git *tree* hash of `benchmark/tasks/heldout/` has neither problem: it is
content-addressed over every byte of the split and nothing else, so it survives later commits
to documentation or reward code, and it changes the instant a task is edited, added or
deleted. The sealing commit SHA is recorded above for provenance, in the commit immediately
after the seal itself.

**The check fails on both routes, verified rather than assumed:**

- recorded tree ≠ committed tree → `HELD-OUT SEAL BROKEN ... has changed since it was sealed`;
- committed tree matches but the working copy is dirty → also fails, because a graded run
  against uncommitted edits would not be scoring the sealed split.

Both were exercised by tampering and reverting. The failure message says explicitly that the
fix is to restore the split, **not** to update `heldout_seal.json` — editing the recorded hash
destroys the only evidence that the held-out numbers were measured on a split nobody had tuned
against.

Contamination is worth enforcing in code precisely because it is silent: nothing errors, the
tasks still validate, the rewards still look plausible, and every generalisation claim
afterwards is a training-set claim wearing a held-out label.

**`docs/A5000_CARRY_FORWARD.md` written** — everything specified and committed here that has
**never executed on this machine**, with the exact Linux check for each:

1. `ResourceLimits.enforced()` — four `setrlimit` caps have never constrained a process
   (`resource` is POSIX-only; `enforced()` reports all four `False` on Windows).
2. gym transport end to end — `uvloop` has no Windows build, so `app.py` cannot even be
   imported here; only asserted statically via `ast`.
3. Regenerate `results/raw/b3_smoke/rollouts_gym.jsonl` at trace schema **v2** — it is the last
   v1 artefact left.
4. Confirm the v2 reward range `[-1, 1]` survives the trainer's advantage computation. Flagged
   as the highest-risk item: a clamp or `max(0, r)` anywhere in the reward pipeline would
   silently collapse `-1.0` gates onto `0.0`, reinstating exactly the hazard ADR 0006's
   coupling constraint exists to prevent — and the fix would be in the trainer, never by
   raising `GATE_REWARD`.

**Commands run:** `check_heldout_seal.py` (pass, and both failure paths) · `uv run pytest` ·
`ruff check .` · `validate_tasks.py --all --no-run`

**Numbers (measured):** seal check **pass**, `318fedc8868b` unchanged; unit tests **191 passed,
1 skipped**; ruff **0 findings**; validator **36 tasks, 0 errors**.

**No training run. No weights downloaded. No checkpoint exists.**
