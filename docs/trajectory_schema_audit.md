# Trajectory schema audit (Phase 2.2)

Status: **audit stage**. This document compares what `analysis/trace_schema.json` records today
(schema v2) against the field list required by Phase 2.2 of `docs/CURRENT_PROJECT_STATUS.md`,
classifies every gap, and states what schema v3 must add.

The audit is written before the schema is changed, so the diff in `analysis/trace_schema.json`
can be read against a stated requirement rather than against an author's memory.

---

## 1. What a trajectory, a rollout and an episode are

These three words are used loosely in the code today and the looseness is what makes the field
list ambiguous. They are fixed here and used consistently from schema v3 onwards.

| Term | Definition | Cardinality |
|---|---|---|
| **Episode** | One materialised sandbox plus one task, from `EpisodeSession.open()` to `close()`. Owns the working tree, the tool log and the grade. | 1 sandbox |
| **Rollout** | One sampling pass of the policy (or of the scripted-gold agent) through one episode, ending in a submitted patch, a terminated turn budget, or an error. | 1 episode, 1 `rollout_id` |
| **Trajectory** | The recorded sequence of everything the agent observed, decided and received inside one rollout: the ordered steps, their tool calls and results, the candidate patch, the verification summary and the reward. | 1 row in `rollouts.jsonl` |

**The relationship.** A trajectory is the *artifact*; a rollout is the *process* that produced it;
an episode is the *environment* the process ran in. One row of `rollouts.jsonl` is one
trajectory, and it is a complete record of one rollout, which happened inside one episode.

**Why the distinction earns its keep.** GRPO samples `k` rollouts of one prompt. Those `k`
rollouts are `k` trajectories — same task, same episode configuration, different samples.
`rollout_id` is therefore the GRPO group key: if two rows share a `task_id` and a `run_id` but
differ in `rollout_id`, they are members of one GRPO group, and the group's reward variance is
exactly what the gradient is built from. Conflating the three terms is what lets a "group" of
identical rollouts look like a valid group with zero variance, which is the failure mode
recorded in `results/raw/b6_rl_step3_variance_pilot/`.

**Valid termination reasons.** Every trajectory ends for exactly one reason, recorded in
`termination_reason`:

| Value | Meaning |
|---|---|
| `submitted` | The agent called `submit()`. Verification was run against the submitted tree. |
| `max_turns` | The turn budget was exhausted without a submission. |
| `max_tool_calls` | The tool-call budget was exhausted without a submission. |
| `timeout` | The wall-clock or a per-tool timeout ended the rollout. |
| `error` | The harness raised. The trajectory is still persisted, with the error class named. |

`termination_reason` is a *cause*; `result.gates` is the *consequence*. They are not redundant:
`max_turns` and `max_tool_calls` both produce a `no_submit` gate, and only the termination
reason distinguishes them. Today that distinction is lost — see gap G3.

---

## 2. Trajectory-level fields

Requirement source: `docs/CURRENT_PROJECT_STATUS.md` §2.2, first list.

| # | Required field | v2 status | Where it lives in v2 / what is missing |
|---|---|---|---|
| T1 | `run_id` | **PRESENT** | top-level |
| T2 | `task_id` | **PRESENT** | `task.id` |
| T3 | `rollout_id` | **PRESENT** | top-level |
| T4 | `model_name` | **PARTIAL** | `agent.model`; `null` for scripted-gold runs |
| T5 | `checkpoint_id` | **MISSING** | no field. A LoRA checkpoint cannot be identified, so two runs of "the 1.5B model" are indistinguishable. |
| T6 | `git_commit` | **PARTIAL** | `versions.repo_commit` |
| T7 | `random_seed` | **MISSING** | no field. Sampling is not reproducible from the row. |
| T8 | `prompt_config_version` | **MISSING** | no field. A prompt edit silently changes the dataset with no trace. |
| T9 | `start_time` / `end_time` | **PRESENT** | `started_at` / `finished_at` |
| T10 | `termination_reason` | **MISSING** | only *inferable* from `result.gates`, and ambiguously so (gap G3) |
| T11 | `total_turns` | **MISSING** | no field. `budgets.tool_calls_used` counts calls, not turns. |
| T12 | `total_tool_calls` | **PRESENT** | `budgets.tool_calls_used` |
| T13 | `token_usage` | **MISSING** | no field. Cost and context pressure are not measurable. |
| T14 | tool-call sequence | **PRESENT** | `tool_calls[]`, ordered by `index` |
| T15 | tool results | **PRESENT** | `tool_calls[].output` (truncated, length preserved in `output_chars`) |
| T16 | candidate patch | **MISSING** | no field. The submitted diff is not recoverable from the row — the sandbox is deleted. |
| T17 | submission status | **PRESENT** | `result.submitted` |
| T18 | verification summary | **PRESENT** | `result.per_test`, `result.passed`, `result.collected` |
| T19 | reward components | **MISSING** | only the fused `result.reward` exists. reward_v2 needs `A/B/C/D` stored separately. |
| T20 | final reward | **PRESENT** | `result.reward` |

**Score: 9 present, 3 partial, 8 missing.**

### Why T16 is the most serious gap

The sandbox is deleted when the episode closes. `tool_calls[].arguments` is clipped at
`MAX_FIELD_CHARS = 600`, so a `write_file` of a source file is truncated. `apply_patch` arguments
are likewise clipped. That means **the submitted patch cannot be reconstructed from a v2 row**,
and a reviewer cannot answer "what did the model actually change?" for any run already on disk.
This is the one gap that is unrecoverable after the fact, which is why v3 records a
`candidate_patch` object (the unified diff, clipped, plus its true length and a SHA-256).

---

## 3. Step-level fields

Requirement source: `docs/CURRENT_PROJECT_STATUS.md` §2.2, second list.

Today there is **no step concept** in the schema. `tool_calls[]` is a flat list of dispatch
outcomes: it records that a tool was called and what it returned, but not what the model
observed before deciding, nor how long the call took, nor what it cost.

| # | Required field | v2 status | Note |
|---|---|---|---|
| S1 | `step_id` | **MISSING** | index is per-call, not per-turn |
| S2 | `timestamp` | **MISSING** | no per-step timing at all |
| S3 | observation summary | **MISSING** | the tool *output* is stored, but not the observation the model actually received |
| S4 | model action | **MISSING** | the decision (which tool, which arguments) is inferable from the call, but the action *type* is not recorded |
| S5 | tool name | **PRESENT** | `tool_calls[].name` |
| S6 | tool arguments | **PRESENT** | `tool_calls[].arguments` (clipped) |
| S7 | tool result summary | **PRESENT** | `tool_calls[].output` (clipped) |
| S8 | success / error | **PRESENT** | `tool_calls[].ok` |
| S9 | duration | **MISSING** | no per-call timing, so slow tools cannot be found |
| S10 | token count | **MISSING** | no per-call cost |

**A turn is not a call.** One turn is one model generation, which may emit zero or one tool
calls. A model that emits prose and no call still consumes a turn. `total_turns` (T11)
therefore cannot be derived from `len(tool_calls)`, and a `steps[]` array keyed on turns is the
structure that makes the distinction explicit.

---

## 4. Leakage requirements

Requirement source: `docs/CURRENT_PROJECT_STATUS.md` §2.3.

| # | Requirement | v2 status | Note |
|---|---|---|---|
| L1 | Agent-visible JSON separated from private verification JSON | **NOT IMPLEMENTED** | one row carries both |
| L2 | Trajectory records only a hidden-test summary | **VIOLATED** | `result.per_test` stores `nodeid -> outcome` for every hidden test, in the same object the dashboard reads |
| L3 | Hidden-test error detail never returned to the model | **HOLDS BY CONSTRUCTION** | verifier truncates to a count; no nodeid or traceback reaches the tool output. Needs a regression test so it stays true. |
| L4 | Dashboard and logs cannot accidentally expose the gold patch | **NOT IMPLEMENTED** | `task.gold_patch()` is read during grading; nothing prevents a future log line from including it |
| L5 | A leakage regression test exists | **PARTIAL** | `benchmark/tests/test_leakage.py` covers the corpus; there is no test on the trace row |

**L2 is a genuine tension, not a bug to be deleted.** `per_test` is described in the schema as
"what makes a failure taxonomy possible later", and that is true — the analysis needs per-test
outcomes. The problem is that the *same* object is what a dashboard or a log would dump. v3
therefore splits the row into two artifacts:

- `rollouts.jsonl` — agent-visible and analysis-safe. Carries a **summary**:
  `hidden_tests: {declared, passed, floor, failed_names_redacted_count}`. No nodeids.
- `rollouts_private.jsonl` — carries `per_test` nodeid->outcome and the full patch, written
  alongside but never read by anything that renders to the agent or to a live dashboard.

Both share `(run_id, rollout_id)`, so a join is always possible for analysis, and the private
file is never the thing that gets pasted into a slide.

---

## 5. Datasets on disk: are failed trajectories persisted?

Requirement: *guarantee failed trajectories are also persisted* (§2.2).

| Observation | Evidence |
|---|---|
| `rollouts_failures.jsonl` is present but empty in every run directory | `results/raw/*/rollouts_failures.jsonl` — 0 bytes across the board |
| Several `rollouts.jsonl` are 0 bytes | `results/raw/b6_rl_step3_variance_pilot/` and the three `*_group_8` runs |
| No `error` row exists anywhere on disk | no row has a termination reason at all, because the field does not exist |

The empty failure file is consistent with `write_jsonl` being called with an empty iterable when
a rollout raises: the writer creates the file, writes nothing, and the exception propagates. The
trajectory that would have explained the crash is the one trajectory that is never recorded.

**This is gap G5 and it is the direct cause of the research blocker.** A group whose rollouts all
crashed scores `mean/reward 0.0, std/reward 0.0` — indistinguishable from a group that ran fine
and achieved nothing, because the crashed rows are absent rather than marked. v3 must write an
`error`-terminated row for every raised rollout.

---

## 6. What schema v3 adds

Ordered by the requirement it closes. Every addition is a new key, so `additionalProperties:
false` is satisfied by editing the schema rather than by relaxing it.

**New top-level keys**

- `termination_reason` — enum, five values (§1). Closes T10, G3.
- `turns` — `{max_turns, turns_used}`. Closes T11.
- `token_usage` — `{prompt, completion, total}`, all nullable for scripted-gold. Closes T13.
- `candidate_patch` — `{present, sha256, chars, truncated, diff}`. Closes T16, G1.
- `reward_components` — `{correctness, progress, validity, tool, final}`; `null` for components
  not yet shipped, so a v1 row is honestly labelled rather than back-filled. Closes T19.
- `steps[]` — ordered, one per turn. Closes S1–S4, S9, S10, G2.
- `hidden_tests` — the redacted summary (§4 L2). Closes L2 partially, by redaction.
- `error` — `{type, message}` or `null`. Closes G5 alongside `termination_reason: "error"`.

**New keys inside existing objects**

- `agent.checkpoint_id`, `agent.random_seed`, `agent.prompt_config_version` — all nullable for
  scripted-gold. Closes T5, T7, T8.

**Deliberately not added**

- `task.gold_patch` or any reference that resolves to it — see L4.
- Per-test nodeids in the agent-visible row — see L2.
- Any field that can be derived by joining two existing fields.

---

## 7. Gap register

| ID | Gap | Severity | Closed by |
|---|---|---|---|
| G1 | Submitted patch unrecoverable (sandbox deleted, arguments clipped) | **High** | `candidate_patch` |
| G2 | No step/turn structure; turns not countable | **High** | `steps[]`, `turns` |
| G3 | No termination reason; `no_submit` conflates three causes | **High** | `termination_reason` |
| G4 | Checkpoint, seed and prompt version unrecorded | **Medium** | `agent.*` |
| G5 | Failed rollouts not persisted; empty failure files | **Critical** | `error` + writer change |
| G6 | Reward components fused into one float | **Medium** | `reward_components` |
| G7 | `per_test` nodeids visible on the agent-facing row | **High** | `hidden_tests` redaction + private file |
| G8 | Token usage unrecorded | **Low** | `token_usage`, `steps[].token_count` |

**G5 is Critical** because it is the only gap that has already destroyed data: the runs that
would explain the zero-variance groups are the runs whose rows are missing.

---

## 8. Migration

`additionalProperties: false` means v3 rows are not valid v2 rows and vice versa. The loader
must therefore:

1. Read `schema_version` before validating, and validate with the matching schema.
2. Keep the v2 schema in the repository, not just in git history, so rows written before this
   change remain readable.
3. Never re-write a v2 row as v3 by inventing values. A v2 row is honestly "no termination
   reason recorded"; back-filling `submitted` from `result.submitted` would fabricate a field
   the run never wrote.

The existing behaviour of refusing unknown keys is preserved and is what makes (3) enforceable.
