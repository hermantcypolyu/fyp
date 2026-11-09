# Phase 6L — Controlled Model-Capacity Diagnostic (1.5B → 3B)

**Status:** DIAGNOSTIC ONLY — not a Phase 6 sub-phase, not training evidence, and it does not open Phase 7.
**Run ID:** `phase6l_bugfix_offbyone_pagination_3b_20261109T102513Z`
**Written:** 2026-11-09
**Code commit:** `536325b` (config + tests committed and pushed **before** execution)
**Category:** **C** — the 3B model does not materially improve over the 1.5B control.

---

## 1. Why this diagnostic exists

Phase 6I, 6J and 6K progressively fixed everything that could be fixed without touching the model:

| Phase | Change | Outcome |
|---|---|---|
| 6I | baseline pilot | uniform −1.0, no usable workflow |
| 6J | workspace-path remediation | paths correct, still no submission |
| 6K | hard context budget + workflow guidance | infrastructure clean, still uniform −1.0 |

After 6K the prompt-level remediation surface was **closed**: the agent was reading the correct path, the
context budget was hard-enforced, and workflow guidance was presented correctly, yet rewards remained a
constant floor. The remaining untested variable was **model capacity**.

Phase 6L therefore changes exactly one thing — model size — and holds every other lever fixed.

---

## 2. The single intentional variable

**Model size: 1.5B → 3B.** Nothing else changed.

| Lever | Control (6K) | Treatment (6L) | Same? |
|---|---|---|---|
| Model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `Qwen/Qwen2.5-Coder-3B-Instruct` | **no — intentional** |
| Revision | `2e1fd397ee46e1388853d2af2c993145b0f1098a` | `488639f1ff808d1d3d0ba301aef8c11461451ec5` | **no — intentional** |
| Adapter | none | none | yes |
| Quantisation | bf16 | bf16 | yes |
| Context window | 8192 | 8192 | yes |
| Task | `bugfix_offbyone_pagination` | `bugfix_offbyone_pagination` | yes |
| Rollouts | 4 | 4 | yes |
| Concurrency | 1 | 1 | yes |
| Temperature | 1.0 | 1.0 | yes |
| top_p | 1.0 | 1.0 | yes |
| Base seed | 42 | 42 | yes |
| Seed policy | `per_rollout` / `sha256_u32_v1` | `per_rollout` / `sha256_u32_v1` | yes |
| Seeds | `184288048, 3168762815, 475852200, 1693580453` | identical | yes |
| Workflow guidance | 6K revision | identical | yes |
| Workspace guidance | 6K revision | identical | yes |
| Context hard limit | enforced | enforced | yes |
| Max turns | `from_task_yaml` (20) | `from_task_yaml` (20) | yes |
| Max tool calls | `from_task_yaml` (40) | `from_task_yaml` (40) | yes |
| Reward | `reward_v1` live, `reward_v2` offline | identical | yes |
| Held-out | disabled | disabled | yes |
| Optimizer / checkpoints | none | none | yes |

**Tokenizer parity was verified before execution.** The 1.5B and 3B tokenizers produce bit-for-bit
identical token ids on the same inputs, with the same `vocab_size` (151643), the same chat template and the
same special tokens. The context budget is therefore a genuinely controlled variable rather than an
accidentally changed one.

### GPU override (explicitly authorised)

The project rulebook (`docs/PHASE5_GPU_RESOURCE_RULES.md`) designates GPU 0 and states that if GPU 0 is
occupied by a foreign process the answer is to wait or ask — **not** to silently move to GPU 1.

GPU 0 was held by foreign PID `2506406` (45196 MiB, ~100% utilisation) for the whole window. A one-off run
on physical **GPU 1** was explicitly authorised by the operator for this run only. It is recorded in
`server.cuda_visible_devices: "1"`, asserted by
`test_the_gpu_override_is_recorded_rather_than_implicit`, and the foreign process was never touched.

Observed GPU state around the run:

| Moment | GPU 0 | GPU 1 |
|---|---|---|
| before launch | 45212 MiB, 100% | 6 MiB, 0% |
| server up | 45212 MiB, 93% (foreign) | 19036 MiB, 0% |
| after shutdown | 45212 MiB, 88% (foreign) | **6 MiB, 0%** |

Foreign PID `2506406` was still present with 45196 MiB after shutdown.

---

## 3. Execution

- Manifest `executable: true`, `execution_claim: true`, `execution_blockers: []` — required before launch.
- `executed.settings` recorded 3B name, revision `488639f1…`, `cuda_visible_devices: 1`, 4 rollouts.
- Server: vLLM `0.6.1.post2`, `--max-model-len 8192 --gpu-memory-utilization 0.4`, health `200`,
  `/v1/models` reported `Qwen/Qwen2.5-Coder-3B-Instruct` at `max_model_len: 8192`.
- Exactly **4** rollouts executed; `rollouts_failures.jsonl` is empty (0 bytes).
- No optimizer, no `backward()`, no `optimizer.step()`, no checkpoint, no training loop.

The collector exited **2** with the uniform-reward guard:

```
ERROR: all 4 reward-eligible rollouts scored from [-1.0] distinct reward value(s); 6.2 requires >= 2.
```

This is the **designed** behaviour, not a harness fault: artifacts were persisted before the guard fired,
and `--allow-uniform` was deliberately not used. The guard is correct — a constant reward gives GRPO no
gradient, so declaring this a usable training group would be false.

---

## 4. Results

### 4.1 Headline comparison

| Metric | 6K (1.5B, control) | 6L (3B, treatment) |
|---|---|---|
| Infrastructure-valid rows | 4 / 4 | **4 / 4** |
| Parseable tool-call rows | 2 / 4 | **3 / 4** |
| Correct-path calls | 21 | **41** |
| Successful file reads | 20 | **18** |
| Successful edits | 0 | **18** |
| `run_tests` calls *attempted* | 0 | **1** |
| `run_tests` calls *that succeeded* | 0 | **0** |
| Non-empty submissions | 0 | **2** |
| Solved rollouts | 0 | **0** |
| Reward distribution | `{-1.0: 4}` | `{-1.0: 4}` |
| Distinct reward values | 1 | **1** |
| Reward stdev | 0.0 | **0.0** |
| Mean turns | 20.00 | 20.00 |
| Mean tool calls | 5.25 | **10.75** |
| Context compactions | 7 | 6 |
| Peak prompt tokens | 5301 | 6230 |
| Distinct trajectory signatures | 3 | **4** |
| Terminations | `max_turns` ×4 | `max_turns` ×4 |
| Wall clock (sum) | 89.6 s | 155.3 s |
| Transport health | clean | **80 POSTs, 80× 200, 0 4xx/5xx** |

Every infrastructure row is valid; the comparison is therefore **meaningful**, not invalidated by harness
failure. See §6 for why that still does not make it category A.

### 4.2 Per-rollout detail (6L)

| # | Seed | Steps | Tools used | Successful edits | Patch chars | Gates | Termination |
|---|---|---|---|---|---|---|---|
| r00 | 184288048 | 3 | read_file → edit_file → run_tests | 1 | 394 | `no_submit`, `test_count_mismatch` | `max_turns` || r01 | 3168762815 | 20 | 11× read_file, 5× edit_file (all failed) | 0 | 0 | `no_submit`, `test_count_mismatch` | `max_turns` |
| r02 | 475852200 | 0 | **none** | 0 | 0 | `no_submit`, `test_count_mismatch` | `max_turns` |
| r03 | 1693580453 | 20 | list_files, 2× read_file, 18× edit_file | 18 | 669 | `no_submit`, `syntax_broken`, `test_count_mismatch` | `max_turns` |

Note that all four rows carry `no_submit` even though r00 and r03 produced non-empty candidate patches —
the agent authored a patch into the workspace but never issued the submit action, so the verifier never
saw it. `submitted` is `false` on every row.

### 4.3 What the two patches actually contain

This matters, because "2 non-empty submissions" is the only superficially encouraging number in the table.

The gold fix is a ceiling division plus an explicit empty-collection case:

```python
if total_items == 0:
    return 1
# Round up: a trailing partial page still needs a page of its own.
return (total_items + per_page - 1) // per_page
```

**r00** produced:

```python
pages = total_pages(count, per_page + (1 if count % per_page != 0 else 0))
```

This does not fix `total_pages`; it inflates `per_page` at the call site instead. The empty-collection
requirement is untouched, and the arithmetic is wrong for non-multiples — it changes which slice is read
rather than how many pages are reported. It is a plausible-looking, semantically incorrect change.

**r03** produced 669 chars of mangled output: the model attempted a `_calculate_total_pages` method but
emitted the line `if count <= 0:` **sixteen times consecutively** with no body, producing syntactically
broken Python (hence the `syntax_broken` gate).

So the 18 "successful edits" are successful *tool operations*, not successful *fixes*. The edit tool
faithfully applied whatever it was given, including sixteen duplicated broken lines.

---

## 5. What the 3B model genuinely did better

These are real, measured improvements and should not be dismissed:

1. **It engaged with the tool loop.** 10.75 mean tool calls versus 5.25, and 3 of 4 rows parseable versus
   2 of 4. The 1.5B model spent most of its turns re-reading.
2. **It discovered the edit tool.** 18 successful edits versus 0. In 6K the 1.5B model made exactly one
   edit attempt in four rollouts; the 3B model edits readily.
3. **It reached the test runner.** 1 `run_tests` call versus 0 across all prior phases. It also revealed a
   genuine environment defect: `No module named pytest` in the serving interpreter — the run_tests tool
   cannot work in that environment regardless of model quality.
4. **It carried context further.** Peak prompt 6230 tokens with 6 compactions, still within the hard limit,
   and 0 context-overflow rows.

## 6. Why this is still category C

The improvements are **behavioural, not remedial**. Criterion A requires usable within-task reward
variation *and* at least one valid non-empty submission. Neither holds:

- **Reward variation: none.** All four rewards are exactly `-1.0`; stdev is `0.0`; distinct values is 1.
  This is identical to 6I, 6J and 6K.
- **Valid submission: none.** Both non-empty patches fail on the merits — one is semantically wrong, one
  is syntactically broken. Neither would pass even if the submit action had been issued.
- **Zero solved rollouts.** `tests_passed` is 0/0 on every row, against a floor of 6.

Criterion B ("improves workflow behaviour but still no usable reward variation") is the closest fit on
behaviour alone, but B is framed as a *partial success toward* usable reward. Here the behavioural gain
does not translate into any movement on the reward surface at all, and the one apparent gain —
non-empty patches — is contradicted by the patch contents on inspection. Calling this B would overstate
the result. **C is the honest reading: no material improvement over 1.5B.**

### The deeper signal

Three of four rollouts hit `max_turns` having never submitted. Even r00 — which found the bug's
neighbourhood — burned its run without issuing `submit`. The binding constraint is no longer model
capacity and is not (after 6K) prompt wording or context budget. It is that the task's success condition
depends on an action sequence the agent does not reliably complete: edit → verify → **submit**.

At 3B the model can now read, edit and run tests. It still does not close the loop. That is consistent
across 1.5B and 3B, which is why another capacity step is unlikely to change it.

---

## 7. Recommendation

**No further model-capacity step is justified on this evidence, and no 7B arm will be run.**

Reasoning:

- The failure mode is **stable across two model sizes** with a clean controlled comparison. Nothing in the
  6L evidence suggests the reward surface is reachable by scaling alone.
- 7B costs roughly 2.3× the VRAM of 3B at the same window, which on this box conflicts with the foreign
  tenant on GPU 0 and forces another override.
- If capacity were the limiter, 3B should have moved *some* reward. It moved many tool calls and zero
  reward points.

If the goal is a usable training group, the next experiment is **not** another model size — it is either:

1. **Fix the `run_tests` environment** (`pytest` missing in the serving interpreter). This is a verifiable
   infrastructure defect that blocks the verify step for every model, and 6L is the first phase that
   surfaced it. **Done in Phase 6M** — see `docs/PHASE6M_TEST_RUNNER_CORRECTION.md`.
2. **Re-examine task design** — whether a 4-rollout single-task group can ever yield reward variation
   when the task is binary pass/fail with a submit-gated verifier.

Model capacity is a **closed** question for this project. It was tested once (1.5B → 3B) and is not to be
retested at any larger size.

---

## 8. Evidence

Everything is under `results/raw/phase6l_bugfix_offbyone_pagination_3b_20261109T102513Z/`:

| File | Contents |
|---|---|
| `merged_config.yaml` | exact execution config (sha256 `2632886f…`) |
| `manifest.json` | `execution_claim: true`, `execution_blockers: []` |
| `rollouts.jsonl` | 4 public rows |
| `rollouts_private.jsonl` | 4 private rows, mode `0600` |
| `rollouts_failures.jsonl` | 0 bytes |
| `collection_summary.json` | seed plan, plan digest `e6d774b5…`, model identity |
| `vllm_server.log` | 80 POSTs, all 200 |
| `server.pid` | `2484538` (the real server, not the setsid launcher) |

Reproduce:

```bash
make check
.venv/bin/python -m scripts.check_heldout_seal
```

Held-out seal at time of writing: `held-out sealed 2026-08-03: 10 tasks, tree 318fedc8868b unchanged`.

## 9. Explicit non-claims

- No reward code, verifier, task, prompt, tool definition, seed, temperature, `top_p` or budget was changed.
- No held-out task was used; `tasks.split` stayed `train`, `heldout_enabled` stayed `false`.
- No LoRA adapter was loaded; no `backward()`; no `optimizer.step()`; no checkpoint was created.
- **Phase 7 was not started.** This diagnostic does not constitute approval for training.
