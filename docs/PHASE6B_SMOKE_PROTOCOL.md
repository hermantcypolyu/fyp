# Phase 6B smoke protocol — prepared, NOT executed

**Status: PREPARED ONLY.** Nothing in this document has been run. Phase 6B was scoped as
a code-and-test task: no vLLM server, NeMo Gym, NeMo RL, Ray, Docker, GPU job, model load,
rollout collection, training or evaluation was started, and no existing rollout row was
modified. This file exists so that the next session can execute the smoke without
re-deriving it, and so that the order of checks is reviewable before any GPU is touched.

## 0. What the smoke is for

The Phase 6.2 batch lost 8 of 48 rollouts to a model-server HTTP 400. The cause was not an
over-long initial prompt; it was prompt **growth** across 13 turns of assistant prose with
zero tool calls. The smoke exists to prove the new preflight/compaction path prevents that
specific failure on a real server — nothing more. It is not a baseline run and produces no
numbers anyone may compare against anything.

Read `results/raw/phase6_baseline_1p5b/RUN_STATUS.json` first. It records the evidence
this protocol is designed to answer.

## 1. Preconditions (all must hold before starting)

| Check | Command | Expected |
|---|---|---|
| Working tree clean of stray edits | `git status --short` | Only the Phase 6B files, nothing staged |
| Tests pass | `make check` | `held-out sealed`, 36 tasks validated, all tests pass, ruff clean |
| Tokenizer reachable from the serving venv | see §2 | `tokenizer_ready: true` |
| GPU 0 free | `nvidia-smi` | GPU 0 unused; **GPU 1 belongs to another user — never touch** |
| Port free | `ss -ltnp \| grep 8010` | No listener |

`make check` runs `pytest`, `ruff check .` and `ruff format --check .`. It does **not**
start anything on a GPU.

## 2. Tokenizer availability check (CPU only)

The project venv deliberately has no `transformers`; the tokenizer lives in the serving
venv. Confirm the pinned path resolves *before* starting a server, because the collector
fails closed and a wrong path wastes a server start:

```bash
/home/hermanyeung/venvs/qwen-server/bin/python -c "
from analysis.context_budget import load_tokenizer
import yaml
cfg = yaml.safe_load(open('configs/eval/baseline_phase6.yaml'))
tok = load_tokenizer(cfg['context']['tokenizer_path'], revision=cfg['model']['revision'])
print('tokenizer ready:', type(tok).__name__)
"
```

Expected: `tokenizer ready: Qwen2TokenizerFast`.

> **Environment caveat.** The collector runs under `.venv` and imports
> `analysis.context_budget`, which imports `transformers`. `transformers` is not installed
> in `.venv`, so `collect()` will raise `TokenizerUnavailableError` unless the run is
> started with the serving venv on `PYTHONPATH` or `transformers` is installed in `.venv`.
> Resolve this deliberately before running the smoke; do not let the collector fall back to
> a character-count estimate, which is exactly the guess Phase 6.2 was built on.

## 3. Start the server (GPU 0 only)

```bash
CUDA_VISIBLE_DEVICES=0 /home/hermanyeung/venvs/qwen-server/bin/vllm serve \
  Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --revision 2e1fd397ee46e1388853d2af2c993145b0f1098a \
  --served-model-name Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --max-model-len 8192 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.4 \
  --port 8010
```

`--max-model-len 8192` must equal `context.max_model_len` in the config. A probe against a
different window measures a different model.

Confirm the server is up and is the model the config names, then stop; do not collect yet:

```bash
.venv/bin/python scripts/collect_baseline_rollouts.py --config configs/eval/baseline_phase6.yaml --dry-run
```

Expected in the dry-run output:

- `"served_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct"`
- `"revision": "2e1fd397ee46e1388853d2af2c993145b0f1098a"`
- `"tokenizer_ready": true` (and `tokenizer_error: null`)
- `"prompt_capacity_tokens": 7104` (8192 − 1024 reserved − 64 margin)
- `"diagnostic_only": false`

If `tokenizer_ready` is false, stop and fix §2. Do not proceed.

## 4. Targeted overflow reproduction

The Phase 6 failure was one task, `testupdate_money_api_v2`, on turn 13 with 7291
completion tokens and zero tool calls. The cheapest smoke is therefore that single task,
with two rollouts:

```bash
.venv/bin/python scripts/collect_baseline_rollouts.py \
  --config configs/eval/baseline_phase6.yaml \
  --diagnostic-only \
  --task-ids testupdate_money_api_v2 --rollouts 2
```

`--task-ids` is required here, not `--tasks 1`. The overflowing task is **last** in
`run.task_subset` (the first is `bugfix_offbyone_pagination`), so a count-based selection
would exercise the wrong task and silently prove nothing. `--task-ids` refuses any id that
is not in `run.task_subset` before contacting the server, so a typo fails immediately
rather than collecting an empty batch.

`--diagnostic-only` is deliberate: the smoke is expected to fail the run-level guards, and
the flag keeps the batch labelled non-training instead of aborting mid-collection.

**What to look for**, in `results/raw/<run_id>/rollouts.jsonl`:

1. No row has a `BadRequestError` / HTTP 400. This is the pass condition.
2. Where a rollout would have overflowed, either:
   - a `context_budget` object shows `peak_prompt_tokens` near but under 7104 with
     `compaction_events > 0`, or
   - the row terminates with `infrastructure_error` and an `infrastructure_failure` object
     whose `stage` is `preflight` or `preflight_after_compaction` — meaning the harness
     caught it before the server did.
3. Every `infrastructure_error` row has `result.reward: null` and
   `infrastructure_failure.reward_eligible: false`.
4. `collection_summary.json` reports `n_reward_eligible` equal to `n_rollouts` minus
   `n_infrastructure_invalid`, and `baseline_valid` together with a non-empty
   `validity_problems` list if either is non-empty.

A run that passes 1–4 is evidence the correction works. It is still not a baseline.

## 5. Shutdown

Send SIGTERM to the server, wait for exit, and confirm GPU 0 is released:

```bash
kill <vllm-pid>; nvidia-smi --query-gpu=index,memory.used --format=csv
```

The collector installs a SIGTERM handler that exits between rollouts, so a `timeout` guard
stops the batch cleanly rather than mid-episode.

## 6. Explicit non-goals

- No training, no GRPO, no NeMo RL, no reward changes.
- No held-out tasks. `scripts/check_heldout_seal.py` must still report the seal unchanged.
- No modification of `results/raw/phase6_baseline_1p5b/`. Those 48 rows are the evidence
  for the diagnosis and must stay exactly as collected.
- No claim that this smoke produces a usable baseline. Whether a clean rerun is a good
  baseline is a separate question: 32 of 40 Phase 6 reward-eligible rollouts reached
  `max_turns` without submitting and scored −1.0, so the reward distribution is nearly
  degenerate for reasons a context budget does not address. Those 32 rows are a *secondary
  behavioural symptom*, not a primary category: primary categories are mutually exclusive
  and first-match, so they split into 24 `no_submission_turns_exhausted` and 8
  `syntax_failure`. The 32 rows do not all belong to `no_submission_turns_exhausted`.
