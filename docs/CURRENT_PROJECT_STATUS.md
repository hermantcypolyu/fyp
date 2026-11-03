# Current Project Status

**Last updated:** 2026-11-03
**Project:** RL of Small Coding Agents with Verifiable Rewards (FYP 2026/27)
**Stage:** Post-F4. Reward redesign and GRPO audit in progress; Phase 7 is the next RL execution stage.

This document is the single source of truth for **what is done, what is in progress, and what is not yet proven**.
It supersedes dated claims inside `README.md`, `PROGRESS.md`, and the demo scripts where they disagree.
`PROGRESS.md` remains the append-only history; this file states the present.

---

## 1. Completed

Verified by a real command, with output stored under `results/` or reproduced by `make check`.

- **Agent tool loop** — seven tools (`list_files`, `read_file`, `write_file`, `edit_file`,
  `replace_lines`, `run_tests`, `submit`), refusals rather than tracebacks, deny-beats-allow.
- **Docker-backed execution** — sandbox materialised with `include_tests=False`; hidden tests
  never exist inside the agent's world.
- **Trajectory logging** — JSONL rows validated on write against `analysis/trace_schema.json`
  (schema v2), persisted for both `direct` and `gym` transports.
- **Private verification** — verifier rebuilds a pristine grading copy from git, then overlays the
  agent's `repo/`; the agent cannot read, write, or influence the hidden suite.
- **Existing reward transport** — `rewards/transport.py` returns a sentinel (`None`) on transport
  failure and never a fabricated `0.0`; `DiscardTracker` aborts past a 2% discard rate.
- **Controlled one-step LoRA optimizer update** — Phase F4, 2026-10-18
  (`docs/PHASE_F4_CONTROLLED_UPDATE.md`).
- **Parameter-change verification** — 392 LoRA tensors changed; delta L2 `0.038865488736174576`.
- **Checkpoint save/reload** — reloaded parameter difference exactly `0.0`.
- **Five-task SWE-Gym Lite baseline pilot** — external sanity check only; **not** a training
  dataset and **not** evidence of improvement.

Measured facts at the last full verification: **303 unit tests passed, 0 skipped**, **36/36** tasks
solvable by gold and failable without it, held-out seal `318fedc8868b` intact, and
`make check` (seal, validate, verify-gold, test, lint) exiting **0** with 0 ruff findings.

---

## 2. In progress

- **Multi-component reward redesign** — `reward_v2` design, component audit, and unit tests.
  `reward_v1` is retained and is not deleted or overwritten.
- **GRPO implementation audit** — determining whether the current training path implements the
  full GRPO objective or only a group-relative reward-weighted update.
- **Phase 7 preparation** — preparing a small, controlled 1.5B GRPO + LoRA run after the reward
  and GRPO checks are complete.
- **1.5B before-and-after evaluation** — protocol fixed, baseline rollouts not yet collected.

---

## 3. Not yet proven

State each of these plainly in the report. None of them is a completed result.

- **Updated model performs better** — F4 shows a gradient flowed, not that coding performance
  improved.
- **Held-out generalisation** — the 10 held-out tasks remain sealed and have not been evaluated.
- **Full GRPO implementation completeness** — the name in use is "controlled GRPO-style LoRA
  update" until the audit proves otherwise.
- **Multi-iteration RL gains** — every multi-task pilot to date shows `mean/reward = 0.0`, i.e. the
  agent reached the measured floor and no further.
- **3B experiment** — not started.
- **Statistical significance** — no confidence intervals computed on real runs.

---

## 4. Research scope (fixed)

### 4.1 Primary research question

> Can multi-component verifiable rewards, combined with GRPO and LoRA, improve the multi-turn
> debugging performance of a small coding agent?

Every experiment must answer this question directly. If an experiment does not, it is out of scope.

### 4.2 Model progression (fixed)

| Model          | Role                                                       | Status                                          |
| -------------- | ---------------------------------------------------------- | ----------------------------------------------- |
| **1.5B** | Reward, GRPO, and closed-loop validation                   | primary working model                           |
| **3B**   | Main final experiment                                      | gated on 1.5B success                           |
| 7B             | Optional extension, **not** a graduation requirement | existing runs are external baseline pilots only |

Hardware: the A5000 (24 GB) / A6000 class GPU is the plan of record. **A100 is not required at
this stage.**

The report must explain why 1.5B precedes 3B: the reward and training loop must be validated
cheaply before compute is spent on a larger model.

### 4.3 Agent harness (fixed)

The current custom coding-agent harness is the main experiment.

- The harness is **not** changed during the main experiment.
- Harness, reward, model, and dataset are **not** changed simultaneously.
- Hermes is an **optional extension**, not part of the main line.
- Unverified names (e.g. "OpenHands" / "OpenClaw") must not appear in report conclusions until
  confirmed.

### 4.4 Trajectory definition (fixed)

> **A trajectory is the complete ordered sequence of model observations, decisions, tool calls,
> environment responses, and verification outcomes, from task initialisation until submission,
> timeout, or termination.**

The three terms below are used with these meanings throughout the code, the schema, and the
report. They are not interchangeable, and the distinction is load-bearing for GRPO.

| Term                 | Definition                                                                                                                                                                                                      | Cardinality                 |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| **Episode**    | One materialised sandbox plus one task, from `EpisodeSession.open()` to `close()`. Owns the working tree, the tool log and the grade.                                                                       | 1 sandbox                   |
| **Rollout**    | One sampling pass of the policy through one episode, ending in a submission, an exhausted budget, or an error.                                                                                                  | 1 episode, 1 `rollout_id` |
| **Trajectory** | The recorded sequence of everything the agent observed, decided and received within one rollout: the ordered steps, their tool calls and results, the candidate patch, the verification summary and the reward. | 1 row of `rollouts.jsonl` |

A trajectory is the **artifact**, a rollout is the **process** that produced it, and an episode is
the **environment** the process ran in. One row of `rollouts.jsonl` is one trajectory, and it is a
complete record of one rollout that happened inside one episode.

**Why the distinction matters.** GRPO samples $k$ rollouts of one prompt. Those $k$ rollouts are
$k$ trajectories sharing a `task_id` and a `run_id` but differing in `rollout_id` — that triple is
the GRPO group key, and the group's reward variance is exactly what the gradient is built from.
Conflating the terms is what allowed a group of identical or crashed rollouts to be reported as a
valid group with zero variance.

**Valid termination reasons.** Every trajectory ends for exactly one reason, recorded in
`termination_reason`:

| Value              | Meaning                                                                                      |
| ------------------ | -------------------------------------------------------------------------------------------- |
| `submitted`      | The agent called `submit()`; verification ran against the submitted tree.                  |
| `max_turns`      | The turn budget was exhausted without a submission.                                          |
| `max_tool_calls` | The tool-call budget was exhausted without a submission.                                     |
| `timeout`        | The wall clock or a per-tool timeout ended the rollout.                                      |
| `error`          | The harness raised. The trajectory is **still persisted**, with the error class named. |

`termination_reason` records the **cause**; `result.gates` records the **consequence**. They are
not redundant: `max_turns` and `max_tool_calls` both fire the same `no_submit` gate, and only the
termination reason distinguishes them.

---

## 5. Priority ordering

### Must complete

- Multi-component reward (`reward_v2`)
- Clear, formal trajectory definition and schema
- GRPO implementation audit
- 1.5B closed loop
- Baseline vs updated comparison
- Held-out evaluation (once, at the end)
- Report and reproducibility evidence

### Should complete

- 3B LoRA main experiment
- SWE-Gym Lite external evaluation
- Failure analysis
- Confidence intervals

### Optional (only with spare time)

- Hermes harness comparison
- Additional agent frameworks
- Dashboard
- 7B training
- A100 scale-up

### Do not do now

- Switch to 7B immediately
- Borrow an A100 immediately
- Rewrite harness, environment, reward, and trainer at the same time
- Describe a parameter change as a model improvement
- Prioritise a large dashboard over real results
- Keep adding reward components without bound
- Use unverified "OpenClaw" naming
- Touch GPU 1 processes belonging to other users

---

## 6. Roadmap

```
Finalise documents and status
        |
Trajectory definition
        |
Multi-component reward v2
        |
Duplicate metric audit
        |
GRPO implementation audit
        |
1.5B baseline
        |
Phase 7: 1.5B GRPO + LoRA training
        |
Checkpoint reload
        |
Before-and-after evaluation
        |
Held-out evaluation
        |
3B main experiment
        |
SWE-Gym Lite external evaluation
        |
Optional Hermes comparison
        |
Optional 7B extension
```

The immediate priority is to finish the prerequisites and then execute **Phase 7**, rather than
starting a large or uncontrolled GPU experiment:

1. Trajectory definition
2. Multi-component reward v2
3. GRPO audit

Only after these three are complete does the 1.5B closed-loop experiment begin. Phase 7 is the
recommended stage for the actual reinforcement-learning loop: collect multiple rollouts per task,
compute verified rewards and group-relative advantages, update only LoRA parameters, and repeat
for a small 2–5 iteration run. Start with a few training tasks, require reward variation, save
each checkpoint, evaluate briefly after each iteration, and stop if loss, gradients, KL, validity,
checkpoint reload, or data-isolation checks fail. The existing one-step F4 update remains an
optimizer and plumbing validation, not evidence of multi-iteration RL improvement.

---

## 7. Work breakdown by phase

### Phase 0 — Project state and documents

**0.1 Commit the pending presentation documents**

```bash
cd ~/projects/fyp
git diff -- README.md
git diff -- docs/demo/code_walkthrough.md
sed -n '1,240p' docs/demo/cantonese_presentation_script.md
```

Checklist:

- [X] The three documents are accurate
- [X] Stale claims removed (e.g. "no training of any kind has been done")
- [X] "Controlled GRPO-style LoRA update completed" retained
- [X] "Model improvement not yet proven" stated explicitly
- [X] SWE-Gym Lite is not described as a formal RL training dataset
- [X] 7B is not described as the next required model
- [X] Hermes is described as an optional extension

```bash
git add README.md docs/demo/code_walkthrough.md docs/demo/cantonese_presentation_script.md
git diff --cached --check
git diff --cached --stat
git commit -m "docs: align presentation materials with updated FYP plan"
git push origin main
```

**0.2 Status document**

- [X] `docs/CURRENT_PROJECT_STATUS.md` created (this file)
- [X] Linked from `README.md` ("Where to go next")
- [X] Model plan (1.5B → 3B, 7B optional, Hermes optional) stated in `README.md`

---

### Phase 1 — Formalise the scope

- [X] Write the research question into `README.md`
- [ ] Write it into the proposal update / progress report — **outside this repository**
- [X] Write it into the presentation opening
- [X] 1.5B used for reward, GRPO, and closed-loop validation
- [X] 3B used as the final main model
- [X] 7B not a graduation requirement
- [X] A100 not a current hardware requirement
- [X] Report explains the 1.5B → 3B progression
- [X] Existing 7B results described only as external baseline pilots
- [X] Main experiment keeps the current harness
- [X] Hermes placed in optional extension
- [X] Unverified framework names kept out of conclusions
- [X] Any future Hermes comparison keeps model, tasks, reward, and verifier identical

The research question is stated verbatim in `README.md` §"Research question" and
`docs/demo/code_walkthrough.md`; the presentation script states it in Cantonese and repeats the
model progression in its closing section. The scope rules above (harness fixed, Hermes optional,
7B optional, A100 not required, unverified names excluded) are enumerated in §4 of this document.

**Open defect carried forward:** none blocking. `make check` is green as of the lint repair
commit. Keep it green: run `make check` before any claim that verification passed.

---

### Phase 2 — Trajectory definition and data format *(top priority)*

**2.1 Formal definition**

> A trajectory is the complete ordered sequence of model observations, decisions, tool calls,
> environment responses, and verification outcomes from task initialisation until submission,
> timeout, or termination.

- [X] Add the definition to the methodology (§4.4 above)
- [X] Add it to the code walkthrough (`docs/demo/code_walkthrough.md`)
- [X] Add it to the presentation script (`docs/demo/cantonese_presentation_script.md`)
- [X] Separate the terms **trajectory**, **rollout**, and **episode** explicitly (§4.4)
- [X] Define the valid termination reasons (§4.4; enum in `analysis/trace_schema.json`)

**2.2 Trajectory schema**

Every trajectory must record:

`run_id`, `task_id`, `rollout_id`, `model_name`, `checkpoint_id`, `git_commit`, `random_seed`,
`prompt_config_version`, `start_time`, `end_time`, `termination_reason`, `total_turns`,
`total_tool_calls`, `token_usage`, tool-call sequence, tool results, candidate patch, submission
status, verification summary, reward components, final reward.

Every step must record:

`step_id`, `timestamp`, observation summary, model action, tool name, tool arguments, tool result
summary, success/error, duration, token count.

- [X] Audit the existing JSONL fields against the list above (`docs/trajectory_schema_audit.md`; 9 present, 3 partial, 8 missing)
- [X] Identify missing fields (same document, gap register G1–G8)
- [X] Update the schema (`analysis/trace_schema.json` bumped to v3; v2 frozen at `analysis/trace_schema_v2.json`)
- [X] Add schema validation (validate-on-write, dispatched by the row's own `schema_version`)
- [X] Add unit tests (`analysis/tests/test_trajectory_phase2.py`, 18 tests)
- [X] Guarantee a unique ID per trajectory (`run_id` + `task.id` + `rollout_id`; `agent.checkpoint_id`/`random_seed`/`prompt_config_version` complete the identity)
- [X] Guarantee failed trajectories are also persisted (`build_row(result=None, error=...)` writes a valid `termination_reason: "error"` row)

**2.3 Prevent private-data leakage**

The agent must never see: hidden test contents, the gold patch, the reference solution, private
verifier internals, `FAIL_TO_PASS` answers, or `PASS_TO_PASS` answers.

- [X] Agent-visible JSON separated from private verification JSON (`rollouts.jsonl` vs `rollouts_private.jsonl`, joined on `(run_id, rollout_id)`)
- [X] Trajectory records only a hidden-test summary (`hidden_tests` is counts only: declared / passed / floor / failed)
- [X] Hidden-test error detail is never returned to the model (verifier truncates to a count; asserted by test)
- [X] Dashboard and logs cannot accidentally expose the gold patch (the row carries no gold-patch path or content; only `candidate_patch`, which is the agent's own diff)
- [X] A leakage regression test exists (`analysis/tests/test_trajectory_phase2.py`, the `2.3 leakage` block)

---

### Phase 3 — Multi-component reward *(highest priority)*

**Outcome: A and B shipped; C and D deferred; E rejected as a training reward.**
See `docs/reward_v2_design.md` and `rewards/v2.py`.

**3.1 Keep the existing reward as the baseline**

- [X] `reward_v1` implementation retained (`rewards/compose.py`, untouched as component A)
- [X] `reward_v1` tests retained (`rewards/tests/test_compose.py`, unmodified)
- [X] `reward_v2` design document added (`docs/reward_v2_design.md`)
- [X] Both can be computed over the same trajectory for comparison
  (`scripts/reward_distribution.py` scores all 162 historical rollouts under both)
- [X] Historical results are not overwritten (the script is read-only; the report lands in
  `results/analysis/reward_v2_offline.json`)

**3.2 Reward components**

**A. Correctness** — does the final patch fix the problem?

Hidden-test pass fraction, target tests passed, regression tests maintained.

- [X] Correctness carries the highest weight (0.60 vs 0.20)
- [X] Computed only by the private verifier (unchanged `normalised_reward`)
- [X] Hidden tests never exposed to the agent (unchanged verifier contract)

**B. Progress** — did the result improve relative to before the edit?

- [X] Baseline test state recorded (the *first* `run_tests` observation, not 0)
- [X] Final test state recorded (best-seen across all `run_tests` calls)
- [X] Net improvement computed (`smoke_delta`, capped by `MAX_SMOKE_DELTA`)
- [X] Previously passing tests protected from regression (regression is priced by A, not
  double-charged; ending below baseline scores 0 progress)
- [X] Regression penalty decided (progress never goes negative; A carries it)
- [X] **Unblocked — the data was always there.** An earlier revision marked B "blocked,
  because the toolbox does not record per-turn test counts". That was wrong: `run_tests`
  writes `"N/M passed (...)"` into `ToolResult.output`, and the rollout JSONL keeps the full
  tool transcript. `rewards/v2.py::smoke_delta` reads it, and
  `scripts/reward_distribution.py` now recovers progress for every historical row.
- [X] **Measured: B observes zero progress on all 162 historical rollouts.** 95 of them ran
  the smoke suite; none ended with more passing tests than it started with, and only 4 ever
  called `run_tests` more than once. B is correct and measurable — the behaviour it prices
  never occurs. This is a model-behaviour finding, not a reward defect, and it is pinned by
  `rewards/tests/test_progress_recovery.py::test_no_historical_rollout_ever_improved`.

**C. Submission validity** — is the submission valid? *(**deferred** to Phase 4)*

Submitted; patch applies; syntax correct; non-empty change; no out-of-scope writes.

- [ ] Valid submission defined
- [ ] Empty / no-op patch defined
- [ ] Invalid patch defined
- [ ] Out-of-scope change defined
- [X] Soft penalty vs hard gate decided per case: **gate**, because every case except
  "empty no-op patch" is already a gate (`no_submit`, `syntax_broken`, `tests_modified`,
  `sandbox_escape`, `invalid_patch`, `forbidden_path_modified`). Pricing them again as a
  weight would score the same failure twice.

**D. Tool behaviour** — did the agent use tools reasonably? *(**deferred** to Phase 4)*

Tool-call format correctness, tool error rate, invalid arguments, repeated no-op operations,
whether the agent responded to failing tests.

- [ ] Only objective behaviour measured
- [ ] No subjective labels (e.g. "smart step")
- [ ] No fixed solution order required
- [ ] Reasonable exploration not penalised

> Deferred because shipping C and D alongside A and B would move four weights at once while
> the model's variance is still zero, and a resulting change in behaviour could not be
> attributed to any one of them. D is also the component most likely to encode a preferred
> *strategy* rather than an outcome.

**E. Efficiency** — what did the task cost? *(**rejected** as a training reward)*

Turns, tool calls, tokens, runtime, repeated file reads, repeated unchanged test runs.

- [X] Efficiency keeps a low weight — **zero**, in the training reward
- [X] Correctness always outweighs efficiency (trivially, since E is absent)
- [X] Premature submission not encouraged — the reason E is rejected: an efficiency term in
  the reward teaches the agent to submit sooner, and the cheapest way to submit sooner is to
  submit without working. The budget is already enforced server-side (`max_tool_calls`) and
  agent-side (`max_turns`), so an efficiency term adds an incentive and no information.
- [X] Efficiency bonus only when correct or valid — moot; it is an **evaluation metric**
  (turns-to-solve, tokens-per-solved-task), not a reward
- [X] Budget-normalised score used — for the metric, not the reward

**3.3 Hard-zero safety gates**

Six are implemented in `envs/coding_env/verifier.py` (`ALL_GATES`) and all return
`GATE_REWARD`. The remaining two are not implemented and are not claimed.

- [X] `no_submit` (`GATE_NO_SUBMIT`)
- [X] `syntax_broken` (`GATE_SYNTAX_BROKEN`, AST-parses every file under `repo/`)
- [X] `tests_modified` (`GATE_TESTS_MODIFIED`, forbidden globs + pytest-hook detection)
- [ ] `private_data_access` — **not implemented**; hidden tests are absent from the sandbox
  rather than guarded against
- [X] `sandbox_escape` (`GATE_SANDBOX_ESCAPE`, resolved-path check in `Toolbox._resolve`)
- [ ] `invalid_patch` — **not implemented**; no patch-apply step exists to fail
- [ ] `forbidden_path_modified` — covered in practice by `tests_modified`'s glob check, but
  exists as no separate gate
- [ ] `malicious_or_prohibited_action` — **not implemented**, and not definable objectively
  without a classifier; deliberately out of scope
- [X] `timeout` and `test_count_mismatch` also exist, beyond this list

> Not every failure should hard-zero. Solving the task incorrectly is low correctness, not a
> safety violation. All six implemented gates are safety/validity failures, not wrong answers.

**3.4 Duplicate metric audit** — `docs/reward_v2_design.md` §5

(No separate `docs/reward_component_audit.md`; the audit lives inside the design document,
where each table sits next to the component it audits.)

For each component:

- [X] What phenomenon does it measure?
- [X] What raw data does it need?
- [X] Does it overlap another component?
- [X] Can the agent manipulate it?
- [X] Is it a **training reward** or an **evaluation metric**?
- [X] What behaviour change follows from reweighting it?

The split between **training reward** and **evaluation metrics** must be explicit. Pass@1, for
example, can be a primary evaluation metric without being injected into every training step.
Recorded as a table in §4 of the design document: pass@1, turns-to-solve, tokens-per-solved
and tool error rate are evaluation metrics and are deliberately absent from the reward.

**3.5 First weight design**

- [X] Design rationale written first (`docs/reward_v2_design.md`, before the code)
- [X] No components added on impulse (E rejected, C and D deferred, with reasons)
- [X] New reward computed offline over existing rollouts (`scripts/reward_distribution.py`,
  162 rows / 53 runs; report at `results/analysis/reward_v2_offline.json`)
- [X] Reward distribution inspected (per-run and pooled mean/std printed and saved)
- [X] Check for a mass of all-zero rewards — **found, and larger than expected**: 25 of 28
  multi-rollout groups have zero spread; 13 of those are all-gated groups no reward can fix
- [X] Check whether all rollouts score nearly the same — 25 groups score *identically*
- [X] Check whether one component dominates the total — correctness does by design (0.60 vs
  0.20); the ceiling on progress's contribution is proven at 0.10 of group spread
  ($\sigma_{final} = 0.20\sigma_B \le 0.10$), verified numerically

Suggested shape: correctness as the main part; progress secondary; validity a precondition or
small score; tool behaviour a small adjustment; efficiency the lowest weight; safety gates zero the
episode.

> **Deviation, stated explicitly.** Validity and tool behaviour are not scored; efficiency is
> scored at zero. The shape above was not adopted wholesale, and the reason is in the design
> document: three of those four are either already gates or are incentives to submit early.

**3.6 Reward unit tests** — `rewards/tests/test_v2.py` (45 tests)

- [X] All tests pass, submission valid (`test_solving_everything_with_perfect_progress_reaches_0_80`)
- [X] Partial improvement (`test_progress_is_measured_against_the_first_observation_not_zero`)
- [X] No improvement (`test_the_unmeasured_case_is_distinguished_from_the_zero_case`)
- [X] Regression introduced
  (`test_a_negative_correctness_with_perfect_progress_is_still_negative`)
- [X] Syntax error (`test_a_gated_run_stays_at_the_bottom_of_the_scale`, parametrised)
- [X] No submit (`test_a_gated_run_stays_at_the_bottom_of_the_scale`, parametrised)
- [ ] Empty patch — **blocked**: no `invalid_patch` or `no_repo_change` policy exists, so
  there is no gate to assert against and inventing one is out of scope for this phase
- [X] Tests modified (`test_a_gated_run_stays_at_the_bottom_of_the_scale`, parametrised)
- [ ] Forbidden file modified — **blocked**: currently subsumed by the `tests_modified` glob
  behaviour and has no separate gate taxonomy to test
- [X] Valid tool calls but wrong answer (`test_progress_alone_cannot_reach_a_solved_rollout_no_progress`)
- [X] Correct answer with heavier tool use (efficiency is absent, so it cannot be penalised;
  asserted structurally by `test_only_the_shipped_components_carry_weight`)
- [X] No double counting between components — each observation feeds exactly one component:
  `test_changing_only_correctness_moves_component_a_and_leaves_progress_alone`,
  `test_changing_only_progress_moves_component_b_and_leaves_correctness_alone`,
  `test_the_two_components_are_the_only_addends_of_the_non_gated_total`,
  `test_one_observation_feeds_exactly_one_component`,
  `test_a_gate_zeroes_the_final_but_preserves_the_diagnostic_components`
- [X] Reward always finite (no `NaN` / `Infinity`) —
  `test_final_is_finite_for_every_admissible_combination` (exhaustive),
  `test_very_large_valid_counts_stay_finite`,
  `test_extreme_progress_inputs_stay_finite_and_bounded`,
  `test_the_degenerate_denominator_is_a_contract_violation_not_a_nan`
- [X] Reward stays within the declared range —
  `test_every_final_value_stays_within_the_declared_outer_range`,
  `test_a_gate_reaches_exactly_the_declared_minimum`,
  `test_the_reachable_maximum_is_the_sum_of_the_shipped_weights`,
  `test_negative_correctness_stays_above_the_declared_minimum`,
  `test_the_reachable_range_is_narrower_than_the_declared_range`
  — declared outer range is `[−1.0, 1.0]`; reachable v2 range is `[−1.0, 0.80]`

---

### Phase 4 — GRPO implementation audit

> **Audited 2026-11-04.** Findings and evidence: `docs/PHASE4_GRPO_AUDIT.md`.
> Verdict: **15 of 20 §4.2 items verified, 1 satisfied by construction, 2 absent
> (reference-policy KL and everything that depends on it), 2 not applicable.**
> The code is a one-step, clipped, group-relative policy-gradient update with LoRA —
> *GRPO-style*, not full GRPO.

**4.1 Locate the training files**

- [X] Rollout adapter — `training/rollout_adapter.py` (12 tests)
- [X] Reward-to-advantage code — `training/grpo_math.py::group_advantages`
- [X] Policy loss — `training/grpo_math.py::policy_loss`, mirrored in the torch loop
- [X] Optimizer step — `scripts/run_optimizer_smoke.py:477`
- [X] Checkpoint save/reload — `scripts/run_optimizer_smoke.py:492`
- [X] LoRA config — `scripts/run_optimizer_smoke.py:301` and `configs/training/*.yaml`
- [X] Training tests — `training/tests/` (39 tests, all torch-free)

**4.2 Verify the GRPO core**

- [X] Multiple rollouts per task — `load_samples` enforces one `task_id`; run used 8
- [X] Independent reward per rollout — `_require_reward` rejects non-finite
- [X] Group mean computed — reproduced the recorded `-0.125` exactly
- [X] Group-relative advantage computed — `(r - mean) / (std + eps)`, bit-identical
- [X] Zero-variance groups handled — `DegenerateGroupError`, never a silent zero
- [X] Generated-token log probabilities obtained
- [X] Prompt tokens masked — logits sliced from `prompt_length - 1`
- [X] Padding tokens masked — **by construction**: turns are scored unpadded, one
  sequence each; no batched path exists to pad
- [X] Old-policy probability (or reference) available — `generation_log_probs`
- [X] Policy probability ratio computed — `policy_ratio`
- [X] Clipped policy objective present — one-sided `min(r*A, clip(r)*A)`, as PPO specifies
- [ ] **Reference-policy KL regularisation present — ABSENT.** No reference model is
  loaded; `grep -n "kl\|ref_model"` over the script returns nothing. Deferred because a
  frozen 1.5B copy does not fit: the run already peaked at 19.9-22.9 GB of 24 GB.
  `test_there_is_no_kl_term_to_test` pins the absence so this cannot be ticked silently.
- [ ] **KL direction correct — not applicable**, since no KL term exists.
- [X] Loss applied only to generated tokens
- [X] Backward produces finite gradients — recorded norm 0.307842467
- [X] Gradient clipping sane — `max_norm=1.0`, did not bind (0.3078)
- [X] AdamW updates only LoRA parameters — 18,464,768 of a 1.5B model
- [X] Base model parameters frozen — ⚠️ evidenced by the recorded run, **not** a CI test
- [X] Parameters actually change after the step — 392 LoRA tensors, delta L2 0.038865 ⚠️
- [X] Checkpoint reload restores identical parameters — `reload_max_parameter_difference: 0.0` ⚠️

> ⚠️ marks items that need torch + CUDA to test and are therefore evidenced only by a
> recorded run, not by a CI test. Do not upgrade these to a stronger claim.

**4.3 Fixed naming**

- [X] "GRPO with LoRA" only if the full GRPO objective is verified — it is **not**
  verified (no KL), so this name must not be used
- [X] Otherwise "Controlled GRPO-style LoRA update" — this is the name in use in
  `README.md`, `docs/PHASE_F4_CONTROLLED_UPDATE.md`, `docs/PHASE_F3_OPTIMIZER_SMOKE.md`,
  `docs/TRAINING_READINESS.md` and `PROGRESS.md`. No rename was needed.
- [X] README, report, and slides use the same name — a repository-wide grep finds no
  competing claim
- [X] A smoke test is never described as full training — `README.md:13` states the update
  "verifies optimizer and checkpoint plumbing only"
- [X] A parameter change is never described as performance improvement —
  `README.md:13` and `PHASE_F4_CONTROLLED_UPDATE.md` §"Interpretation and limitations"
  both say it does **not** show improved coding performance

**4.4 New GRPO tests** — `training/tests/test_grpo_math.py` (27 tests, torch-free)

- [X] Equal rewards → advantage near zero — **rejected instead**, deliberately:
  `test_equal_rewards_are_rejected_rather_than_scored_as_zero_advantage`. A zero-variance
  group is `0/0`; returning `0.0` would look like a successful step with no gradient.
- [X] High-reward rollout → positive advantage — `test_high_reward_rollout_gets_positive_advantage`
- [X] Low-reward rollout → negative advantage — `test_low_reward_rollout_gets_negative_advantage`,
  `test_the_lowest_reward_always_gets_the_lowest_advantage`
- [X] Clipping actually bounds the ratio — `test_clipping_limits_the_reward_for_improving_too_far`,
  `test_clipping_does_not_apply_to_the_disadvantageous_direction`,
  `test_inside_the_trust_region_the_objective_is_not_clipped`
- [ ] **Rising KL is reflected in the loss — cannot be tested; no KL term exists.**
  `test_there_is_no_kl_term_to_test` pins the gap with `inspect.signature`.
- [X] Prompt tokens excluded from policy loss — `test_loss_is_normalised_by_the_group_token_count_not_the_turn_count`
- [ ] **Padding tokens excluded from policy loss — not applicable.** Each turn is scored as
  one unpadded sequence of `len(prompt) + len(generation)`; there is no batched path.
- [ ] Frozen base weights unchanged — ⚠️ needs a GPU; evidenced by the recorded run's
  `trainable_parameter_count`, not by a CI test
- [ ] LoRA weights changed — ⚠️ needs a GPU; `parameter_delta` is torch-bound
- [X] Loss, gradient, KL all finite — `test_loss_is_finite_for_every_extreme_ratio_and_advantage`,
  `test_policy_ratio_output_is_always_finite` (loss and ratio finite; KL absent)
- [X] Empty / invalid rollout safely rejected — `test_mismatched_lengths_are_rejected`,
  `test_non_finite_log_probabilities_are_rejected`,
  `test_zero_token_count_is_rejected_rather_than_dividing_by_zero`,
  `test_empty_inputs_are_handled_without_a_divide_by_zero`, plus the 12 adapter tests

**Anchor test.** `test_reproduces_the_recorded_smoke_run_exactly` replays the rewards from
an executed RTX A6000 run and asserts the mean, standard deviation and both advantage values
to the last decimal. That is the evidence that extracting the math into `training/grpo_math.py`
did not change it.

---

### Phase 5 — Fixed experiment protocol

**5.1 Record the 1.5B configuration**

- [ ] Base model name and revision
- [ ] LoRA rank
- [ ] LoRA alpha
- [ ] Target modules
- [ ] Learning rate
- [ ] Group size
- [ ] Number of tasks
- [ ] Rollouts per task
- [ ] Temperature
- [ ] Top-p
- [ ] Max turns
- [ ] Max tool calls
- [ ] Max sequence length
- [ ] Gradient clipping
- [ ] KL coefficient
- [ ] Clip epsilon
- [ ] Random seeds
- [ ] Training iterations
- [ ] Checkpoint schedule

**5.2 Run manifest** — `run_id`, date, git commit, model, dataset version, reward version,
trajectory schema version, training config hash, evaluation config hash, input checkpoint, output
checkpoint, GPU, seed, result folder.

- [ ] A manifest is generated per run
- [ ] Config hash recorded
- [ ] Checkpoint hash recorded
- [ ] GPU model recorded
- [ ] Start and end time recorded
- [ ] Failure reason recorded
- [ ] Old results never manually overwritten

**5.3 GPU resource rules**

- [ ] Use GPU 0 only
- [ ] Every command sets `CUDA_VISIBLE_DEVICES=0` explicitly
- [ ] GPU 1 processes belonging to other users are never touched
- [ ] vLLM stopped after rollouts
- [ ] GPU freed after training
- [ ] `nvidia-smi` checked before leaving
- [ ] No idle server holding VRAM long-term
- [ ] Server PID and log path recorded
- [ ] Reasonable timeouts set

---

### Phase 6 — 1.5B baseline evaluation

**6.1 Fix the baseline**

- [ ] Original 1.5B base model
- [ ] No LoRA checkpoint loaded
- [ ] Prompt fixed
- [ ] Harness fixed
- [ ] Tool descriptions fixed
- [ ] Sampling settings fixed
- [ ] Turn / tool budget fixed
- [ ] Verifier fixed
- [ ] `reward_v2` fixed
- [ ] Evaluation task list fixed

**6.2 Collect enough rollouts**

- [ ] Multiple rollouts on each training task
- [ ] More than one task
- [ ] Avoid identical rewards across all rollouts
- [ ] Successful trajectories saved
- [ ] Failed trajectories saved
- [ ] Gated trajectories saved
- [ ] Tool errors saved
- [ ] Tokens and runtime recorded

**6.3 Baseline analysis**

- [ ] Mean reward
- [ ] Reward standard deviation
- [ ] Pass@1
- [ ] Hidden-test success rate
- [ ] Valid-submission rate
- [ ] Syntax-failure rate
- [ ] Safety-gate rate
- [ ] Mean turns
- [ ] Mean tool calls
- [ ] Mean token usage
- [ ] Mean runtime
- [ ] Failure categories

---

### Phase 7 — 1.5B GRPO + LoRA training

This is the recommended stage to perform the actual RL experiment. It must remain a small,
controlled feasibility run before scaling to 3B: use a few training tasks, multiple rollouts per
task, fixed prompts and harness settings, and only 2–5 update iterations initially. Do not
increase the task count, model size, or training duration until reward variation, finite losses
and gradients, checkpoint integrity, and basic evaluation behaviour are acceptable.

**7.1 First formal small run**

- [ ] Few training tasks
- [ ] Multiple rollouts per task
- [ ] Reward variation confirmed
- [ ] Group advantage sane
- [ ] Loss finite
- [ ] Gradient finite
- [ ] KL finite
- [ ] Base model frozen
- [ ] LoRA parameters updated
- [ ] Checkpoint saved
- [ ] Optimizer state saved if needed
- [ ] Full logs saved

**7.2 Safety stop conditions** — stop the run if any of:

- [ ] Loss becomes `NaN`
- [ ] Gradient becomes `NaN` or explodes
- [ ] KL spikes
- [ ] All rewards identical
- [ ] Safety-gate rate suddenly rises
- [ ] Valid submissions clearly drop
- [ ] Checkpoint fails to reload
- [ ] Base model updated unexpectedly
- [ ] GPU OOM or abnormal resource use
- [ ] Private data leakage

**7.3 Controlled iterations**

- [ ] Second rollout → verify → update
- [ ] Third iteration
- [ ] Decide 2–5 iterations based on results
- [ ] Checkpoint saved each iteration
- [ ] Small evaluation each iteration
- [ ] Never pad training steps using zero-variance rewards

---

### Phase 8 — 1.5B before-and-after evaluation *(most important evidence)*

**8.1 Fair comparison** — identical across both arms: tasks, prompt, harness, tools, sampling
settings, turn limit, tool-call limit, verifier, reward version, seeds/seed protocol, evaluation
budget. The only difference is **base 1.5B vs updated 1.5B + LoRA checkpoint**.

- [ ] All of the above held constant and recorded

**8.2 Primary metric**

- [ ] Primary metric: Pass@1 or task success rate
- [ ] Success condition formally defined
- [ ] Training reward never substituted for evaluation success
- [ ] One official Pass@1 result per task, or the repetition strategy stated clearly

**8.3 Secondary metrics**

- [ ] Hidden-test pass rate
- [ ] Mean multi-component reward
- [ ] Valid-submission rate
- [ ] Regression rate
- [ ] Syntax-failure rate
- [ ] Safety-gate rate
- [ ] Average tool calls
- [ ] Average turns
- [ ] Token usage
- [ ] Runtime
- [ ] Tool-error rate

**8.4 Failure analysis** — classify into: wrong diagnosis, incorrect file selected, incomplete fix,
regression introduced, invalid tool call, syntax failure, no submission, premature submission,
timeout, safety-gate violation, repetitive or inefficient behaviour.

**8.5 Conclusion rules**

*Improved:*

- [ ] Report the size of the improvement
- [ ] Report which task types improved
- [ ] Check for safety or efficiency cost

*No clear change:*

- [ ] Inspect the reward signal
- [ ] Inspect the number of iterations
- [ ] Inspect task difficulty
- [ ] Inspect LoRA capacity
- [ ] Report an honest negative result

*Regressed:*

- [ ] Inspect KL
- [ ] Inspect learning rate
- [ ] Inspect reward hacking
- [ ] Inspect overfitting
- [ ] Inspect reward-component conflict

---

### Phase 9 — Held-out evaluation

Only after the training pipeline is stable.

- [ ] The 10 held-out tasks have never participated in training
- [ ] Held-out contents not accidentally read
- [ ] Fixed evaluation protocol
- [ ] Baseline vs updated checkpoint compared
- [ ] Pass@1 computed
- [ ] Paired per-task difference computed
- [ ] Confidence interval computed
- [ ] Generalisation analysed
- [ ] Sample-size limitation reported
- [ ] No strong statistical claim from only 10 tasks

---

### Phase 10 — 3B main experiment

Start only when: `reward_v2` validated; trajectory schema stable; GRPO implementation audited;
1.5B checkpoint reloads; before-and-after protocol works; held-out evaluation runs; GPU 0 safe; the
experiment command is reproducible.

- [ ] 3B baseline built
- [ ] Same reward
- [ ] Same harness
- [ ] Same task split
- [ ] Batch / group size adjusted for VRAM
- [ ] LoRA used
- [ ] One 3B optimizer smoke run first
- [ ] Then a small controlled training run
- [ ] Before-and-after comparison
- [ ] Scaling effect compared against 1.5B

---

### Phase 11 — SWE-Gym Lite external evaluation

External validity only.

- [ ] Official task format confirmed
- [ ] Repositories and base commits confirmed
- [ ] Dependencies confirmed
- [ ] Verifier mapping confirmed
- [ ] Agent-visible data isolated
- [ ] Existing five-task baseline retained
- [ ] Fixed checkpoint used for external evaluation
- [ ] External evaluation tasks never reused for training
- [ ] Internal benchmark and real-repository tasks compared
- [ ] Reported as a pilot only, unless the task count is sufficient

---

### Phase 12 — Optional: Hermes harness

**Optional. Not main line.**

- [ ] Hermes parser vs Hermes agent harness distinguished
- [ ] NeMo Gym version confirmed to include a Hermes adapter
- [ ] Hermes verified to work with the current model endpoint
- [ ] Hermes verified to use the same sandbox and verifier
- [ ] Trajectory format mapping confirmed
- [ ] Fair comparison keeps model, checkpoint, tasks, reward, verifier, sampling, and budget
  identical — only the harness changes
- [ ] If time is short, Hermes moves to Future Work rather than delaying the main evaluation

---

### Phase 13 — Optional: 7B / A100

Not now. Consider only if: 1.5B and 3B main experiments are complete; the report already has enough
results; there is spare time; GPU resources are stable; a specific research question needs 7B;
other Polyverse users are not affected; an A6000 is demonstrably insufficient; the supervisor
agrees that scale-up is worthwhile.

Otherwise, describing 7B as a future extension is sufficient.

---

### Phase 14 — Parallel final-report writing

Do not wait for all experiments to finish before writing.

**Methodology** — updated architecture; formal trajectory definition; multi-component reward
formula; hard-zero gates; GRPO vs LoRA division of labour; agent harness role; NeMo Gym environment
role; model progression 1.5B → 3B; dataset split; data-leakage prevention; evaluation protocol.

**Results** — pipeline engineering validation; reward distribution; training curves; gradient
evidence; parameter-change evidence; checkpoint integrity; baseline vs updated; held-out results;
failure analysis; external benchmark results.

**Limitations** — small task count; hand-authored benchmark; limited training iterations; LoRA
rather than full-parameter update; shared GPU constraints; reward-weight sensitivity;
harness-specific behaviour; 7B not formally trained; limited statistical power.

---

## 8. Immediate next actions (top 10, in order)

1. Review and commit the three pending presentation documents.
2. Create the status document separating completed / in progress / not yet proven. **(done)**
3. Write the formal trajectory definition and schema.
4. Preserve `reward_v1` and design `reward_v2`.
5. Perform the duplicate-metric audit.
6. Add unit tests and leakage tests for `reward_v2`.
7. Audit the training code to classify it as full GRPO or GRPO-style.
8. Fix the 1.5B baseline and experiment configuration.
9. Collect baseline rollouts, then execute the Phase 7 small 1.5B GRPO + LoRA run.
10. Reload the updated checkpoint and complete the fair baseline vs updated 1.5B comparison in Phase 8.

---

## 9. Related documents

- `docs/TRAINING_READINESS.md` — inventory and reasoning
- `docs/RUN_ON_A5000.md` — ordered runbook, and the "failures that look like valid scores" hazards
- `docs/A5000_CARRY_FORWARD.md` — what has never executed
- `docs/PHASE_F3_OPTIMIZER_SMOKE.md`, `docs/PHASE_F4_CONTROLLED_UPDATE.md` — measured optimizer evidence
- `docs/decisions/0006-normalise-reward-against-the-measured-floor.md` — reward normalisation
- `docs/decisions/0007-multi-turn-rollout-adapter-and-unscorable-rollouts.md` — adapter contract
- `PROGRESS.md` — append-only build history
