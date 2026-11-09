# Phase 6M — Test-Runner Environment Correction and Diagnostic Rerun

**Status:** DIAGNOSTIC ONLY — not training evidence, does not open Phase 7.
**Run ID:** `phase6m_bugfix_offbyone_pagination_3b_20261109T104748Z`
**Written:** 2026-11-09
**Code commit:** `14eaf8f` (fix + tests committed and pushed **before** execution)
**Category:** **C** — the model does not materially improve; the *harness* defect is fixed.

---

## 1. What this Phase is

Phase 6M contains exactly two things, and nothing else:

1. A narrow correction to the test runner, because Phase 6L proved the verify step was
   impossible for every model.
2. One rerun of the 6L diagnostic arm under byte-identical conditions, to measure what the
   harness fix changed.

It is **not** a capacity experiment, **not** a reward change, and **not** a new sub-phase of the
remediation series.

---

## 2. Root cause, proven

`run_tests` invoked pytest with `sys.executable`. Rollouts are launched by
`scripts/run_in_serving_env.py` under the **serving** venv
(`/home/hermanyeung/venvs/qwen-server`), so inside a rollout `sys.executable` *is* the serving
interpreter — which carries vLLM and transformers but deliberately omits the project's dev
extras.

`pytest` is declared by this repository as a **dev** dependency:

```
[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.6"]
```

So the serving venv was never going to have it. The empirical proof, run against the very same
task:

| Interpreter | `collected` | `passed` | `collection_error` | Summary |
|---|---|---|---|---|
| serving venv (pre-fix) | 0 | 0 | `True` | `COLLECTION ERROR` |
| project `.venv` (pre-fix) | 3 | 2 | `False` | `2/3 passed` |

The serving-interpreter stderr was literally:

```
/home/hermanyeung/venvs/qwen-server/bin/python: No module named pytest
```

Phase 6L's single `run_tests` call failed with exactly this text, and the agent was told:

```
Error: smoke tests could not be collected - the repo probably no longer imports.
```

That message blamed the candidate's patch for an environment fault. This is why **no arm in the
entire series** — 6I, 6J, 6K or 6L — ever recorded a successful `run_tests`.

---

## 3. The correction

Three changes, all in the shared task-loader path that both evaluation and RL reward use
(`AGENT_BUILD_SPEC` 5.6 invariant: one canonical runner).

1. **`_resolve_test_interpreter()`** resolves, in order: `FYP_TEST_PYTHON` (if set and
   executable) → the repo's `.venv/bin/python` → `sys.executable` as a last-resort fallback.
   `run_tests` now builds its argv as `[interpreter, "-m", "pytest", ...]`.
2. **`TestResult.infrastructure_error`** is a new field. A missing pytest is detected from the
   interpreter's own stderr and classified as infrastructure, summarised as
   `INFRASTRUCTURE ERROR` ahead of `TIMEOUT` and `COLLECTION ERROR` in the precedence chain.
3. **The agent-facing message** now says the runner is unavailable and explicitly states this is
   not caused by the edit, so the model is not misled into rewriting correct code.

**Why not install pytest into the serving venv.** That venv exists to serve models with pinned
versions (`vLLM 0.6.1.post2`, `transformers 4.45.2`, `tokenizers 0.20.3`); mutating it would
invalidate the pinned-version record the experiment depends on. The repository declares pytest as
a dev extra, so test execution belongs to the project interpreter. Serving and testing processes
stay independent by construction.

---

## 4. Evidence the fix is safe

- `make check` green: **908 passed, 1 skipped**, ruff clean, 139 files formatted.
- Held-out seal unchanged: `held-out sealed 2026-08-03: 10 tasks, tree 318fedc8868b unchanged`.
- **22 new CPU-only tests** in `analysis/tests/test_phase6m_test_runner_interpreter.py` covering
  the seven required proofs: the project interpreter is used; pytest imports; exit codes are
  preserved; counts are parsed correctly; missing pytest is an infrastructure failure and not a
  candidate defect; the serving interpreter is not mutated; and the Phase 6K/6L evidence
  directories are untouched.
- Phase 6K and 6L evidence directories were **not** modified. 6L's failure text is deliberately
  preserved: evidence is a historical record and the fix is not retro-applied to it.

---

## 5. The rerun is a controlled comparison

`configs/eval/phase6l_model_capacity_3b.yaml` was reused **unchanged**. The materialised run
config differs from 6L's only in run identity:

```
166c166
<   analysis: results/analysis/phase6l_bugfix_offbyone_pagination_3b_20261109T102513Z.json
---
>   analysis: results/analysis/phase6m_bugfix_offbyone_pagination_3b_20261109T104748Z.json
```

Everything else — model, revision `488639f1…`, task, four seeds, `temperature` 1.0, `top_p` 1.0,
prompts, tool definitions, turn and tool budgets, `max_model_len` 8192 — is identical by
construction, not by assertion.

| Item | Value |
|---|---|
| Model | `Qwen/Qwen2.5-Coder-3B-Instruct` |
| Revision | `488639f1ff808d1d3d0ba301aef8c11461451ec5` |
| Task | `bugfix_offbyone_pagination` |
| Rollouts | 4 |
| GPU | physical GPU 1 (documented override; GPU 0 held by a foreign PID) |
| Server PID | `2779264` (the real process holding port 8010) |
| Manifest | `execution_claim: true`, `execution_blockers: []` |
| Reward config | `v1` live, `v2` offline only |
| Training | none — no adapter, no optimizer, no checkpoint |

---

## 6. The eight required metrics

All figures below were extracted from `results/raw/<run_id>/rollouts.jsonl` with the resolved
path printed, after the 6I/6J extraction defect described in §8.

| # | Metric | 6L (before fix) | **6M (after fix)** |
|---|---|---|---|
| 1 | Successful `run_tests` executions | **0** | **2** |
| 2 | `run_tests` calls that parsed test counts | **0** | **2** |
| 3 | Edits applied successfully | 18 | **19** |
| 4 | Non-empty submissions (patch chars > 0) | 2 | **2** |
| 5 | Reward distribution | `{-1.0: 4}` | **`{-1.0: 3, 0.0: 1}`** |
| 6 | Reward standard deviation | 0.0 | **0.433** |
| 7 | Solved rollouts | 0 | **0** |
| 8 | Infrastructure-invalid rows | 0 | **0** |

Supporting counts (attempts, not successes): `run_tests` attempted 1 → 2; edits attempted
23 → 24.

### 6.1 Per-rollout detail (6M)

| # | Steps | `run_tests` | Edits attempted | Patch chars | Reward | Submitted | Gates |
|---|---|---|---|---|---|---|---|
| r00 | 6 | 2 (both ok) | 2 | 575 | **0.0** | **true** | — |
| r01 | 20 | 0 | 0 | 0 | −1.0 | false | `no_submit` |
| r02 | 0 | 0 | 0 | 0 | −1.0 | false | `no_submit` |
| r03 | 20 | 0 | 17 | 669 | −1.0 | false | `no_submit`, `syntax_broken`, `test_count_mismatch` |

Terminations: r00 stopped on `submitted` after 6 of 20 turns; r01 and r03 hit `max_turns`; r02
produced no steps at all. r00 is the only row in the series that ever terminated by submitting
rather than by exhausting its budget.

The two successful `run_tests` calls are genuine. r00's first call returned
`2/3 passed (smoke tests only; these are not the grading tests)` — the same 2/3 the project
interpreter returns on the unmodified repo — and its second, after editing, returned
`3/3 passed`. The agent therefore observed real feedback and iterated correctly on it.

### 6.2 What changed, and what did not

**Changed.** r00 went from 3 steps and a failing `run_tests` to 6 steps, two successful test
runs, a submitted patch scoring `6/11` grading tests with **no gates**, and reward `0.0`. This is
the first non-floor, non-empty submission in the entire Phase 6 series.

**Did not change.** 0 of 4 rollouts solved the task. Three of four rows still never submitted.
The reward signal is still sparse: three quarters of the group sit at the floor. The 0.0 row is a
*partial* fix on a binary pass/fail task, not a solution.

**On 6L's non-empty patches.** 6L's `edits_ok` of 18 and 6M's 19 are both dominated by r03's
degenerate loop emitting `if count <= 0:` with no body. r00's patch is the one that matters. Both
arms make the *same* core mistake — rewriting the call site as
`pages = total_pages(count, per_page + (1 if count % per_page != 0 else 0))`, which inflates
`per_page` instead of fixing `total_pages`. 6M's r00 additionally rewrote the out-of-range branch
to return `items[0:per_page]`, still behind the wrong `page > pages` guard. Neither patch is
correct; the gold fix is `if total_items == 0: return 1` plus
`return (total_items + per_page - 1) // per_page`.

The difference between the arms is not patch quality but *feedback*: in 6M the model could run
the tests, saw `2/3` then `3/3` on the smoke suite, submitted, and still scored only `6/11` on
the grading suite. That gap is explainable and worth recording. The three smoke tests only check
that the package imports, that a first page of a short list works, and that query chaining works.
None of them touches the three behaviours this task is actually about —
`test_total_pages_rounds_up_when_there_is_a_remainder`,
`test_final_partial_page_is_reachable`, and
`test_empty_collection_still_has_exactly_one_page`. So the model received *true but insufficient*
feedback: a green smoke run is not evidence that the off-by-one bug is fixed.

---

## 7. Interpretation

Not category A, and not category B:

- **Not A (the fix does not solve the task).** 0 of 4 solved; the reward group is not usable for
  GRPO. `min_valid_rollouts` is unmet because three rollouts share the floor value.
- **Not B (the fix is not cosmetic).** It is a real, proven infrastructure defect. Successful
  `run_tests` executions went from 0 to 2 across the whole series, reward variation appeared for
  the first time (stdev 0.0 → 0.433), and one rollout produced a scoring submission with no gates.

The honest reading: the verify loop now **works**, and the model **still cannot fix the bug**.
Those are separate findings, and only the first was the goal of this Phase.

**Decision: C — no material improvement over 6L, but the harness defect is closed.**

---

## 8. Defects found in earlier extraction

Two reporting errors were corrected, both in derived numbers rather than in evidence:

1. **6I/6J metrics extracted from the wrong directories.** Four `phase6i_*` directories exist and
   **none** of them contains `rollouts.jsonl`; a glob-based extractor therefore resolved to the
   parent `raw/` directory. The real arms are
   `phase6i_…20261109T084300Z` and `phase6j_…20261109T090714Z`. Re-extracted with explicit paths
   and a refusal guard, the correct figures are:

   | Arm | `run_tests` attempted / ok | Edits attempted / ok | Rewards | Infra-invalid |
   |---|---|---|---|---|
   | 6I | 0 / 0 | 7 / **0** | `{-1.0: 4}` | 0 |
   | 6J | 0 / 0 | 25 / **0** | `{-1.0: 4}` | 1 |
   | 6K | 0 / 0 | 1 / 0 | `{-1.0: 4}` | 0 |

   The 6I prose figures ("20 file-not-found, 7 edit calls, 0 `run_tests`, 1 empty submission") and
   the 6J figures are **confirmed correct**; the 6J infrastructure-invalid count is confirmed at
   **1**. What was wrong was a derived "edits" column that counted only *successful* edits and so
   reported 0 for arms that demonstrably attempted 7 and 25 — all of which failed, 6I on a
   non-writable path and 6J on `old_str` mismatches.

2. **6L's "`run_tests` 0 → 1"** conflated attempts with successes. The single call **failed**;
   the report now states attempts and successes separately, and the "successful" line reads 0.

---

## 9. Reproduction

```bash
make check
.venv/bin/python -m scripts.check_heldout_seal

.venv/bin/python -m scripts.materialise_eval_config \
    --overlay configs/eval/phase6l_model_capacity_3b.yaml \
    --out results/raw/<run_id>/merged_config.yaml --run-id <run_id>

.venv/bin/python -m scripts.run_manifest --arm baseline --run-id <run_id> \
    --eval-config results/raw/<run_id>/merged_config.yaml --execution-claim -o results/raw/<run_id>

CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/run_in_serving_env.py \
    scripts/collect_baseline_rollouts.py --manifest results/raw/<run_id>/manifest.json \
    -- --config results/raw/<run_id>/merged_config.yaml
```

---

## 10. Explicit non-claims

- No model, task, prompt, tool definition, reward, seed, temperature, `top_p`, budget or verifier
  logic was changed.
- No held-out task was used; `heldout_enabled` stayed `false`.
- No adapter, no `backward()`, no `optimizer.step()`, no checkpoint.
- **No 7B model was used, and no 7B run is recommended.** Model capacity is closed after a single
  1.5B → 3B comparison.
- **Phase 7 was not started.**

The downstream Phase 7 model-arm decision is recorded in
[ADR-0008](decisions/0008-phase-7-uses-3b-not-1p5b.md).
