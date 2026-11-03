# Current Project Status

**Last updated:** 2026-11-03
**Project:** RL of Small Coding Agents with Verifiable Rewards (FYP 2026/27)
**Stage:** Post-F4. Reward redesign and GRPO audit in progress.

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

| Model | Role | Status |
| --- | --- | --- |
| **1.5B** | Reward, GRPO, and closed-loop validation | primary working model |
| **3B** | Main final experiment | gated on 1.5B success |
| 7B | Optional extension, **not** a graduation requirement | existing runs are external baseline pilots only |

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

| Term | Definition | Cardinality |
|---|---|---|
| **Episode** | One materialised sandbox plus one task, from `EpisodeSession.open()` to `close()`. Owns the working tree, the tool log and the grade. | 1 sandbox |
| **Rollout** | One sampling pass of the policy through one episode, ending in a submission, an exhausted budget, or an error. | 1 episode, 1 `rollout_id` |
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

| Value | Meaning |
|---|---|
| `submitted` | The agent called `submit()`; verification ran against the submitted tree. |
| `max_turns` | The turn budget was exhausted without a submission. |
| `max_tool_calls` | The tool-call budget was exhausted without a submission. |
| `timeout` | The wall clock or a per-tool timeout ended the rollout. |
| `error` | The harness raised. The trajectory is **still persisted**, with the error class named. |

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
1.5B GRPO + LoRA training
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

The immediate priority is **not** another GPU run. It is:

1. Trajectory definition
2. Multi-component reward v2
3. GRPO audit

Only after these three are complete does the 1.5B closed-loop experiment begin.

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

- [x] The three documents are accurate
- [x] Stale claims removed (e.g. "no training of any kind has been done")
- [x] "Controlled GRPO-style LoRA update completed" retained
- [x] "Model improvement not yet proven" stated explicitly
- [x] SWE-Gym Lite is not described as a formal RL training dataset
- [x] 7B is not described as the next required model
- [x] Hermes is described as an optional extension

```bash
git add README.md docs/demo/code_walkthrough.md docs/demo/cantonese_presentation_script.md
git diff --cached --check
git diff --cached --stat
git commit -m "docs: align presentation materials with updated FYP plan"
git push origin main
```

**0.2 Status document**

- [x] `docs/CURRENT_PROJECT_STATUS.md` created (this file)
- [x] Linked from `README.md` ("Where to go next")
- [x] Model plan (1.5B → 3B, 7B optional, Hermes optional) stated in `README.md`

---

### Phase 1 — Formalise the scope

- [x] Write the research question into `README.md`
- [ ] Write it into the proposal update / progress report — **outside this repository**
- [x] Write it into the presentation opening
- [x] 1.5B used for reward, GRPO, and closed-loop validation
- [x] 3B used as the final main model
- [x] 7B not a graduation requirement
- [x] A100 not a current hardware requirement
- [x] Report explains the 1.5B → 3B progression
- [x] Existing 7B results described only as external baseline pilots
- [x] Main experiment keeps the current harness
- [x] Hermes placed in optional extension
- [x] Unverified framework names kept out of conclusions
- [x] Any future Hermes comparison keeps model, tasks, reward, and verifier identical

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

- [x] Add the definition to the methodology (§4.4 above)
- [x] Add it to the code walkthrough (`docs/demo/code_walkthrough.md`)
- [x] Add it to the presentation script (`docs/demo/cantonese_presentation_script.md`)
- [x] Separate the terms **trajectory**, **rollout**, and **episode** explicitly (§4.4)
- [x] Define the valid termination reasons (§4.4; enum in `analysis/trace_schema.json`)

**2.2 Trajectory schema**

Every trajectory must record:

`run_id`, `task_id`, `rollout_id`, `model_name`, `checkpoint_id`, `git_commit`, `random_seed`,
`prompt_config_version`, `start_time`, `end_time`, `termination_reason`, `total_turns`,
`total_tool_calls`, `token_usage`, tool-call sequence, tool results, candidate patch, submission
status, verification summary, reward components, final reward.

Every step must record:

`step_id`, `timestamp`, observation summary, model action, tool name, tool arguments, tool result
summary, success/error, duration, token count.

- [x] Audit the existing JSONL fields against the list above (`docs/trajectory_schema_audit.md`; 9 present, 3 partial, 8 missing)
- [x] Identify missing fields (same document, gap register G1–G8)
- [x] Update the schema (`analysis/trace_schema.json` bumped to v3; v2 frozen at `analysis/trace_schema_v2.json`)
- [x] Add schema validation (validate-on-write, dispatched by the row's own `schema_version`)
- [x] Add unit tests (`analysis/tests/test_trajectory_phase2.py`, 18 tests)
- [x] Guarantee a unique ID per trajectory (`run_id` + `task.id` + `rollout_id`; `agent.checkpoint_id`/`random_seed`/`prompt_config_version` complete the identity)
- [x] Guarantee failed trajectories are also persisted (`build_row(result=None, error=...)` writes a valid `termination_reason: "error"` row)

**2.3 Prevent private-data leakage**

The agent must never see: hidden test contents, the gold patch, the reference solution, private
verifier internals, `FAIL_TO_PASS` answers, or `PASS_TO_PASS` answers.

- [x] Agent-visible JSON separated from private verification JSON (`rollouts.jsonl` vs `rollouts_private.jsonl`, joined on `(run_id, rollout_id)`)
- [x] Trajectory records only a hidden-test summary (`hidden_tests` is counts only: declared / passed / floor / failed)
- [x] Hidden-test error detail is never returned to the model (verifier truncates to a count; asserted by test)
- [x] Dashboard and logs cannot accidentally expose the gold patch (the row carries no gold-patch path or content; only `candidate_patch`, which is the agent's own diff)
- [x] A leakage regression test exists (`analysis/tests/test_trajectory_phase2.py`, the `2.3 leakage` block)

---

### Phase 3 — Multi-component reward *(highest priority)*

**3.1 Keep the existing reward as the baseline**

- [ ] `reward_v1` implementation retained
- [ ] `reward_v1` tests retained
- [ ] `reward_v2` design document added
- [ ] Both can be computed over the same trajectory for comparison
- [ ] Historical results are not overwritten

**3.2 Reward components**

**A. Correctness** — does the final patch fix the problem?

Hidden-test pass fraction, target tests passed, regression tests maintained.

- [ ] Correctness carries the highest weight
- [ ] Computed only by the private verifier
- [ ] Hidden tests never exposed to the agent

**B. Progress** — did the result improve relative to before the edit?

- [ ] Baseline test state recorded
- [ ] Final test state recorded
- [ ] Net improvement computed
- [ ] Previously passing tests protected from regression
- [ ] Regression penalty decided

**C. Submission validity** — is the submission valid?

Submitted; patch applies; syntax correct; non-empty change; no out-of-scope writes.

- [ ] Valid submission defined
- [ ] Empty / no-op patch defined
- [ ] Invalid patch defined
- [ ] Out-of-scope change defined
- [ ] Soft penalty vs hard gate decided per case

**D. Tool behaviour** — did the agent use tools reasonably?

Tool-call format correctness, tool error rate, invalid arguments, repeated no-op operations,
whether the agent responded to failing tests.

- [ ] Only objective behaviour measured
- [ ] No subjective labels (e.g. "smart step")
- [ ] No fixed solution order required
- [ ] Reasonable exploration not penalised

**E. Efficiency** — what did the task cost?

Turns, tool calls, tokens, runtime, repeated file reads, repeated unchanged test runs.

- [ ] Efficiency keeps a low weight
- [ ] Correctness always outweighs efficiency
- [ ] Premature submission not encouraged
- [ ] Efficiency bonus only when correct or valid
- [ ] Budget-normalised score used

**3.3 Hard-zero safety gates**

- [ ] `no_submit`
- [ ] `syntax_broken`
- [ ] `tests_modified`
- [ ] `private_data_access`
- [ ] `sandbox_escape`
- [ ] `invalid_patch`
- [ ] `forbidden_path_modified`
- [ ] `malicious_or_prohibited_action`

> Not every failure should hard-zero. Solving the task incorrectly is low correctness, not a
> safety violation.

**3.4 Duplicate metric audit** — `docs/reward_component_audit.md`

For each component:

- [ ] What phenomenon does it measure?
- [ ] What raw data does it need?
- [ ] Does it overlap another component?
- [ ] Can the agent manipulate it?
- [ ] Is it a **training reward** or an **evaluation metric**?
- [ ] What behaviour change follows from reweighting it?

The split between **training reward** and **evaluation metrics** must be explicit. Pass@1, for
example, can be a primary evaluation metric without being injected into every training step.

**3.5 First weight design**

- [ ] Design rationale written first
- [ ] No components added on impulse
- [ ] New reward computed offline over existing rollouts
- [ ] Reward distribution inspected
- [ ] Check for a mass of all-zero rewards
- [ ] Check whether all rollouts score nearly the same
- [ ] Check whether one component dominates the total

Suggested shape: correctness as the main part; progress secondary; validity a precondition or
small score; tool behaviour a small adjustment; efficiency the lowest weight; safety gates zero the
episode.

**3.6 Reward unit tests**

- [ ] All tests pass, submission valid
- [ ] Partial improvement
- [ ] No improvement
- [ ] Regression introduced
- [ ] Syntax error
- [ ] No submit
- [ ] Empty patch
- [ ] Tests modified
- [ ] Forbidden file modified
- [ ] Valid tool calls but wrong answer
- [ ] Correct answer with heavier tool use
- [ ] No double counting between components
- [ ] Reward always finite (no `NaN` / `Infinity`)
- [ ] Reward stays within the declared range

---

### Phase 4 — GRPO implementation audit

**4.1 Locate the training files**

- [ ] Rollout adapter
- [ ] Reward-to-advantage code
- [ ] Policy loss
- [ ] Optimizer step
- [ ] Checkpoint save/reload
- [ ] LoRA config
- [ ] Training tests

**4.2 Verify the GRPO core**

- [ ] Multiple rollouts per task
- [ ] Independent reward per rollout
- [ ] Group mean computed
- [ ] Group-relative advantage computed
- [ ] Zero-variance groups handled
- [ ] Generated-token log probabilities obtained
- [ ] Prompt tokens masked
- [ ] Padding tokens masked
- [ ] Old-policy probability (or reference) available
- [ ] Policy probability ratio computed
- [ ] Clipped policy objective present
- [ ] Reference-policy KL regularisation present
- [ ] KL direction correct
- [ ] Loss applied only to generated tokens
- [ ] Backward produces finite gradients
- [ ] Gradient clipping sane
- [ ] AdamW updates only LoRA parameters
- [ ] Base model parameters frozen
- [ ] Parameters actually change after the step
- [ ] Checkpoint reload restores identical parameters

**4.3 Fixed naming**

- [ ] "GRPO with LoRA" only if the full GRPO objective is verified
- [ ] Otherwise "Controlled GRPO-style LoRA update"
- [ ] README, report, and slides use the same name
- [ ] A smoke test is never described as full training
- [ ] A parameter change is never described as performance improvement

**4.4 New GRPO tests**

- [ ] Equal rewards → advantage near zero
- [ ] High-reward rollout → positive advantage
- [ ] Low-reward rollout → negative advantage
- [ ] Clipping actually bounds the ratio
- [ ] Rising KL is reflected in the loss
- [ ] Prompt tokens excluded from policy loss
- [ ] Padding tokens excluded from policy loss
- [ ] Frozen base weights unchanged
- [ ] LoRA weights changed
- [ ] Loss, gradient, KL all finite
- [ ] Empty / invalid rollout safely rejected

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
9. Collect baseline rollouts, then run the first multi-component reward training run.
10. Reload the updated checkpoint and complete the fair baseline vs updated 1.5B comparison.

---

## 9. Related documents

- `docs/TRAINING_READINESS.md` — inventory and reasoning
- `docs/RUN_ON_A5000.md` — ordered runbook, and the "failures that look like valid scores" hazards
- `docs/A5000_CARRY_FORWARD.md` — what has never executed
- `docs/PHASE_F3_OPTIMIZER_SMOKE.md`, `docs/PHASE_F4_CONTROLLED_UPDATE.md` — measured optimizer evidence
- `docs/decisions/0006-normalise-reward-against-the-measured-floor.md` — reward normalisation
- `docs/decisions/0007-multi-turn-rollout-adapter-and-unscorable-rollouts.md` — adapter contract
- `PROGRESS.md` — append-only build history
