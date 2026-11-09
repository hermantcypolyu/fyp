# Phase 6K — final behavioural remediation comparison

The third and last experiment in a three-run diagnostic series on the same task,
model, sampling settings and four deterministic seeds:

| Phase | Remediation | Outcome |
| --- | --- | --- |
| 6I | none (control) | model could not find the file |
| 6J | workspace-path guidance | model found the file, never edited it |
| 6K | workflow procedure + hard token limit | model still does not complete the workflow |

Phase 6K is the last planned behavioural remediation before a model-size or
task-design decision. No further Phase 6 sub-phase follows it.

## Why

Phase 6J removed the navigation defect but the reward did not move. Two rollouts
read the target file and then stopped; the rest either repeated identical failed
edit attempts or never called a tool at all. Two hypotheses remained open:

1. The model was running out of usable context. `compact_messages` accepted a
   `budget` and a `tokenizer` and **used neither**, so a "compacted" conversation
   could still exceed `max_model_len`. Phase 6J's `r02` died exactly this way:
   after compaction it still required 9242 tokens against 8192, and the request
   was never sent.
2. The model did not know what sequence of actions constitutes progress. Nothing
   in the prompt said that reading is not the goal.

Phase 6K tests both at once, because they are independent defects with
independent fixes. If neither moves the reward, the remaining explanation is
capacity or task design, not harness instructions.

## What changed

### 1. A hard post-compaction token limit

`analysis/context_budget.py`. The core defect was that `compact_messages`
ignored its own `budget` and `tokenizer` arguments. It now:

- measures the assembled prompt with the tokenizer after every pass;
- escalates a ladder on each failed pass until the prompt fits:
  drop one more retained tool message, then halve `observation_char_cap` down to
  `policy.min_char_cap`, then elide assistant prose entirely;
- terminates after at most 12 passes and reports honestly when it cannot fit.

`CompactionResult` gained `passes`, `within_budget` and `prompt_tokens` so the
caller can distinguish "compacted and fits" from "compacted and still too large".
When no tokenizer or budget is supplied, `within_budget` is `None` rather than a
fabricated pass.

One refinement mattered. Attaching prose to a tool call previously excluded a
message from compaction entirely, so a model narrating 800 characters before
every call carried roughly 10000 tokens of prose that compaction could see and
did not touch. Prose is now clipped or elided while the `tool_calls` payload is
preserved; the call id must stay paired with its tool reply or the server
returns 400.

### 2. Narrow workflow guidance

`scripts/collect_baseline_rollouts.py`, new `WORKFLOW_GUIDANCE` constant appended
to the system prompt after the workspace-path guidance. It names the four phases
in order - read, edit, run_tests, submit - and states that reading is not
progress and that describing a change is not making one.

The guidance is deliberately narrow. It names no tool arguments, no file
contents, no failing test and no reference solution. A parametrised test asserts
twelve leak terms are absent, and that no code fence or `def ` appears in it.

### 3. Peak-prompt telemetry correction

Phase 6J's reported `peak_prompt_tokens` was the turn-0 prompt of the first
rollout, not a maximum. The collector now keeps the probe with the larger
measured prompt and emits `compaction_attempts` alongside `compaction_events`, so
a re-preflight that never produced a compaction event is still visible.

`analysis/trace_schema.json` adds optional `compaction_attempts` only;
`schema_version` stays `"3"` because the change is additive, matching the Phase
6H precedent.

## What deliberately did not change

Reward v1, reward v2, the verifier, the hidden tests, the gold patches, the
model name, revision and size, the task, `temperature` 1.0, `top_p` 1.0, the
base seed 42, the four derived seeds, `max_turns`, `max_tool_calls` and the
compaction character caps. The overlay tracks only the run id and output paths.

Every identity lever was re-read from the materialised config and compared
directly against Phase 6J before the run:

```
model.name                     Qwen/Qwen2.5-Coder-1.5B-Instruct
model.revision                 2e1fd397ee46e1388853d2af2c993145b0f1098a
model.adapter                  None
sampling.temperature           1.0
sampling.top_p                 1.0
sampling.seed                  42
sampling.seed_policy           per_rollout
sampling.seed_derivation       sha256_u32_v1
sampling.max_tokens_per_turn   1024
run.rollouts_per_task          4
run.task_subset                ['bugfix_offbyone_pagination']
context.max_model_len          8192
context.safety_margin_tokens   64
context.reserved_completion_tokens 1024
context.observation_char_cap   2000
context.keep_recent_tool_messages 4
```

The four seeds reproduced exactly: `184288048`, `3168762815`, `475852200`,
`1693580453`, plan digest
`e6d774b5a2752f08f912fd85f968d461b52b6195d0c6cb2187af6d2aa6fa06ec`.

## Result (recorded 2026-11-09)

Run directories:
`results/raw/phase6i_bugfix_offbyone_pagination_20261109T084300Z`,
`results/raw/phase6j_bugfix_offbyone_pagination_20261109T090714Z`,
`results/raw/phase6k_bugfix_offbyone_pagination_20261109T100011Z`.

### Per-rollout behaviour

| rollout | turns | tool sequence | reward | peak prompt | compaction attempts |
| --- | --- | --- | --- | --- | --- |
| `r00` | 20/20 | 20 x `read_file` | -1.0 | 5301 | 6 |
| `r01` | 20/20 | 1 x `edit_file` (failed) | -1.0 | 849 | 0 |
| `r02` | 20/20 | none | -1.0 | 5005 | 1 |
| `r03` | 20/20 | none | -1.0 | 849 | 0 |

`r01` produced an `edit_file` call whose `old_str` matched nothing:
`Error: No match for old_str in repo/src/pagination.py. Read the file and copy
the text exactly.` Its `new_str` was also syntactically broken. It never issued
a second call, so the correction it was given produced no follow-up.

`r02` and `r03` emitted no tool call in 20 turns. Neither is an infrastructure
failure: both are `reward_eligible: true`, `infrastructure_failure: null`, and
the server answered every request. They are 40 assistant turns that the collector
could not read a tool call from.

### Headline metrics

| metric | 6I | 6J | 6K |
| --- | --- | --- | --- |
| rollouts persisted | 4 | 4 | 4 |
| reward-eligible | 4 | 3 | 4 |
| infrastructure-invalid | 0 | 1 | 0 |
| context overflow | 0 | 1 | 0 |
| rows in failures file | 0 | 1 | 0 |
| `read_file` calls | 20 | 16 | 20 |
| edit calls | 7 | 25 | 1 |
| `run_tests` calls | 0 | 0 | 0 |
| submit calls | 1 (empty patch) | 0 | 0 |
| file-not-found observations | 20 | 0 | 0 |
| repeated identical calls | 25 | 37 | 19 |
| peak prompt tokens | 605 | 703 | 5301 |
| compaction attempts | n/a | n/a | 6 |
| compaction events | 0 | 2 | 6 |
| rewards | -1, -1, -1, -1 | -1, -1, -1 | -1, -1, -1, -1 |
| distinct rewards | 1 | 1 | 1 |
| reward stdev | 0.0 | 0.0 | 0.0 |
| solved rollouts | 0 | 0 | 0 |
| wall clock (s) | 45.63 | 76.67 | 89.69 |

Every row in all three runs reports `tests_passed: 0`, `tests_total: 0` and
`pass_fraction: 0.0`. The gate list is `["no_submit", "test_count_mismatch"]` on
11 of 12 rows. The single exception is 6I's `r02`, which did call `submit` and
reports `submitted: true` against an empty patch, so `no_submit` is absent there
and it is the one row counted as a submission. Twelve scored rollouts across
three experiments have produced one distinct reward value.

### Transport health

The server log for the Phase 6K window contains 90 `POST /v1/chat/completions`
requests and zero 4xx or 5xx responses. No request was rejected for length. The
hard token limit did its job: the true peak prompt was 5301 tokens against a
capacity of 7104, `status: "ok"`, 1803 tokens remaining.

## Reading the result

Both changes worked as specified, and neither produced a reward signal.

Change 1 is demonstrated. The 6J overflow at 9242 tokens against 8192 cannot
recur: compaction now measures its own output every pass and escalates until the
prompt fits, and the observed peak left 1803 tokens of headroom. Infrastructure
invalid rows fell 1 to 0 and context overflows fell 1 to 0. The
`compaction_attempts` counter is populated for the first time, which is why the
6J row that died could previously be mistaken for a model failure.

Change 2 is demonstrated as a prompt change and not as a behaviour change. The
guidance is present, ordered, and leak-free, and its tests pin all of that. The
model read it and did not act on it. `run_tests` calls remain 0 in every run of
the series, including the two runs where the model made 7 and 25 edits. The
model does not run tests when it edits, and in 6K it barely edits at all.

The two rollouts with zero tool calls are the interesting new failure. Twenty
turns each at `temperature` 1.0, and not one response contained a call the
collector could parse. This is not a truncated context - `r03` peaked at 849
tokens, 6255 tokens below capacity, with no compaction attempt at all. Given a
prompt that explicitly says to read, edit, test and submit, the model spent its
entire budget emitting assistant text.

Decision A is therefore not met for the third time: 1 distinct reward, stdev 0.0,
0 solved rollouts, 0 non-empty submissions. `baseline_acceptance.passed` is
`false` on the uniform-reward criterion. The run is `diagnostic_only: true` and
`training_ready: false`; it is diagnosis, never training evidence.

## Recommendation

The harness is no longer the limiting factor, and further prompt work is not
indicated. Three experiments now show the instructions taking effect in the
transcript and failing to take effect in the behaviour:

- 6J fixed navigation completely and the reward did not move.
- 6K fixed the context ceiling completely and the reward did not move.
- Across 12 scored rollouts, `run_tests` was called zero times.

Each further instruction experiment would be another variable tested against a
fixed outcome. The two open hypotheses that remain are outside the prompt:

**Model size.** A 1.5B instruct model at `temperature` 1.0 spends its turn budget
on prose rather than tool calls, and when it does call `edit_file` it
reconstructs the target text from memory instead of copying it. Two of four
rollouts emitted no parseable call at all. This is a capability and instruction-
following limit, and it is consistent with the 6I-6K evidence. A larger instruct
model is the next lever.

**Task design.** The task requires a multi-step edit-test-submit loop. A model
that cannot complete that loop in 20 turns produces a constant `-1.0` regardless
of its other abilities, so the task measures the loop rather than the reasoning.
A shorter task, or one whose partial progress is visible to the reward, would
separate "cannot plan" from "cannot edit".

Both should be evaluated before any GRPO run. Neither is a Phase 6 sub-phase and
this report starts neither. Recommendation: test a larger instruct model on the
unchanged task first, because the zero-tool-call rollouts indicate the loop is
not being entered at all, which is a capability question before it is a reward
question.

## Evidence

- Overlay: `configs/eval/phase6k_workflow_procedure.yaml`
- Manifests: `manifest.json` in each run directory, `executable: true`
- Configs: `merged_config.yaml` in each run directory
- Rollouts: `rollouts.jsonl` (public), `rollouts_private.jsonl` (mode 0600)
- Failures: `rollouts_failures.jsonl` empty in 6K
- Summary: `collection_summary.json` in each run directory

The Phase 6I and 6J run directories were not modified.
