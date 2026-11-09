# Phase 6J — workspace-path remediation comparison

A single controlled A/B against the Phase 6I diagnostic smoke. One lever changes: the
system prompt now states the workspace root. Everything else is held identical.

## Why

Phase 6I ran four infrastructure-valid rollouts of `bugfix_offbyone_pagination` and all
four scored `-1.0`. The cause was not sampling, the verifier, or the model. The model
addressed files as `src/pagination.py`, but the sandbox root is the materialised working
copy and only resolves paths under `repo/`. Across the four rollouts there were 20
file-not-found calls, 25 repeated identical failing calls, 0 successful edits and 0
successful `run_tests` calls.

Two prompt defects produced this:

1. The configured system message never named the root, so the `repo/` prefix had to be
   guessed.
2. The worked tool-call example used `"path": "src/app.py"` — an unprefixed path the
   sandbox rejects. The prompt demonstrated the exact format that fails.

The task instruction is not at fault and is unchanged. It names `src/pagination.py`
because it describes the file within the repository; making the tool resolve that
unprefixed form is a tool change, not a prompt change, and would hide the behaviour this
experiment measures.

## What changed

`scripts/collect_baseline_rollouts.py`:

- New `WORKSPACE_PATH_GUIDANCE` block, appended by `_system_prompt` between the tool
  calling convention and the worked example. It states the root is `repo/`, requires the
  prefix, shows the visible tree (`repo/src/pagination.py`, `repo/tests/`), and tells the
  model not to repeat an identical call after a miss.
- The worked example now uses `repo/src/pagination.py` instead of `src/app.py`.

The block says nothing about the bug, the hidden tests, or a reference solution, so a
difference against Phase 6I is attributable to path handling rather than to leaking the
answer.

## What deliberately did not change

- `configs/eval/baseline_phase6.yaml` is untouched; `prompt.version` stays
  `baseline_phase6_v1` and `system_message` is unmodified. The guidance is appended at
  render time, so the trace's prompt hash records it.
- No task instruction, model, revision, seed, temperature, top_p, turn/tool budget,
  verifier or reward change.
- No tool auto-prepends `repo/`, silently repairs an argument, or auto-submits. Test 6 in
  `analysis/tests/test_phase6j_workspace_path_guidance.py` asserts that an unprefixed
  path is still rejected even when the file exists at the prefixed location.

## Control identity (must match Phase 6I)

| Field | Value |
|---|---|
| task | `bugfix_offbyone_pagination` (train) |
| model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` @ `2e1fd397ee46e1388853d2af2c993145b0f1098a` |
| seeds | `184288048, 3168762815, 475852200, 1693580453` |
| temperature / top_p | 1.0 / 1.0 |
| rollouts | 4 |
| max_turns / max_tool_calls | 20 / 40 |
| reward | reward_v1 live, reward_v2 offline |

## Reading the result

Compare the Phase 6J run against Phase 6I on the same twelve metrics: infrastructure-valid
rows, correct-path tool calls, file-not-found calls, repeated identical failed calls,
successful edits, `run_tests` calls, non-empty submissions, rewards, distinct rewards,
reward standard deviation, solved rollouts, mean turns and mean tool calls.

Differing trajectories are not improvement. A change counts only if it moves those
metrics in the direction the remediation predicted: fewer not-found calls, more
correct-path calls, and at least one genuine edit or test run.
