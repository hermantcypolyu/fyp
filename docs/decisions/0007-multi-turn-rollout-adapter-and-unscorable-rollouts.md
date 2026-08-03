# 0007 — The multi-turn rollout adapter, and what a rollout is worth when it cannot be scored

- **Status:** **Accepted** — the sentinel is implemented; the adapter is specified, not built
- **Date:** 2026-08-03
- **Milestone:** B4 (handoff), to be built first on the A5000
- **Affects:** `rewards/transport.py`, `configs/training/grpo_unsloth_a5000.yaml`,
  `docs/RUN_ON_A5000.md`, `docs/A5000_CARRY_FORWARD.md`

## Context

Two findings from reading the real integration sources on 2026-08-03 (not from memory):

1. **The Unsloth × NeMo Gym path is single-turn.** Unsloth's own documentation describes the
   collaboration as enabling "single-turn rollout RL training", and the notebook's reward
   function is `reward_fn(completions, prompts=None, **kwargs)` — one completion string, one
   `/verify` POST. Our episode is not that shape: `seed_session` → up to `max_turns` tool calls
   (`list_files`, `read_file`, `write_file`, `edit_file`, `run_tests`, `submit`) → `/verify` on
   the resulting sandbox. There is no single completion to score.
2. **Every reference implementation defaults a failed verify to `0.0`.** Both of them:

   ```python
   # unslothai/notebooks, NeMo-Gym-Multi-Environment.ipynb
   reward = resp.json().get("reward", 0.0) if resp.status_code == 200 else 0.0

   # NVIDIA-NeMo/Gym, responses_api_agents/verifiers_agent/app.py
   reward = rollout_output.get("reward", 0.0) or 0.0
   ```

Under ADR 0006 that second one is now dangerous in a way it was not before. `0.0` used to be
below every task's floor; it now means *exactly* "achieved nothing" — a legitimate, common,
correct score. A failed `/verify` returning `0.0` is therefore indistinguishable from a real
do-nothing rollout. It enters the group, moves the mean and standard deviation GRPO normalises
against, and raises nothing.

### Sources read

| # | Source |
| --- | --- |
| S1 | NeMo Gym Unsloth tutorial + RL-framework-compatibility (`NVIDIA-NeMo/Gym`, `fern/versions/latest/`) |
| S2 | `unslothai/notebooks` → `nb/NeMo-Gym-Multi-Environment.ipynb` |
| S3 | `NVIDIA-NeMo/Gym` → `responses_api_agents/verifiers_agent/app.py` |
| S4 | `NVIDIA-NeMo/Gym` → `responses_api_models/vllm_model/README.md` |
| S5 | `huggingface/trl` → `docs/source/openenv.md`, "Training on harnesses (opencode)" |

## Decision

### 1. Transport failure returns a sentinel, never a score — **implemented**

`rewards/transport.py`:

- `reward_from_verify_response(...) -> float` **raises** `VerifyTransportError` on a non-200, a
  missing/`null`/non-numeric/`bool`/non-finite `reward`, or a non-object body. It has no default
  and no fallback.
- `score_or_discard(...) -> float | None` returns `None` on failure. **`None` means drop the
  rollout from the group**, not score it zero.
- `DiscardTracker` counts scored vs discarded, exposes `summary()` for per-step logging, and
  `check()` raises `DiscardRateTooHigh` past a threshold (default 2%, min 20 samples).
- `rewards/tests/test_transport.py` pins it, including a 500 that *carries* a reward body.

`None` is not an invention. It is TRL's own contract, from S5 on `rollout_reward_fn`:

> Return a `float`, or return `None` to mark the rollout unscorable so it is dropped from the
> group baseline instead of being counted as a zero.

and on `HarnessRolloutOutcome.env_reward`: "`float | None` … or `None` when the rollout could not
be scored."

**Why a threshold rather than silent dropping.** Discarding is correct per-rollout, but a rising
discard rate is a dead environment, and the rollouts that *did* score are then a biased sample of
it. Training on the remainder is worse than stopping, so the tracker fails loudly.

### 2. The adapter — specified here, built first on the A5000

**A rollout = one session = one GRPO group member.** `num_generations` sessions run per prompt,
each with its own sandbox; the group is the set of their scalar rewards. The episode's many turns
do not become many group members.

**Where the trace is captured.** Not by re-running anything. S5's loop-owning path puts a
*transparent proxy* inside the sandbox that forwards the agent's `/v1/chat/completions` calls to
vLLM and records each turn's token ids and logprobs. When the agent stops, TRL reads the proxy
trace, rebuilds per-turn training rows from the recorded ids, and scores the final workspace with
the session's `verify()`. This matches our design: our verifier already rebuilds the hidden suite
from git and grades the agent's `repo/`, and it must stay the single scoring path (§5.6).

**How token ids and logprobs survive.** Each `TraceEntry` carries `request`, `response`,
`completion_token_ids` and `per_token_logps`. The vLLM server must be started so those exist at
all (S5):

```bash
vllm serve <model> \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --logprobs-mode processed_logprobs \
    --return-tokens-as-token-ids \
    --weight-transfer-config '{"backend":"nccl"}'
```

NeMo Gym reaches the same place by a different route: `vllm_model_for_training.yaml` sets
`return_token_id_information: true`, and `verifiers_agent` threads per-turn tokens through
`_tok_kwargs` into `prompt_token_ids` / `generation_token_ids` / `generation_log_probs` on the
`...ForTraining` message and tool-call variants (S3, S4). Either way the principle is the same
and is the reason the adapter cannot simply concatenate a transcript: **re-tokenising a rendered
conversation is not the same token sequence the policy actually emitted**, and the logprobs would
be a different distribution's.

**Which turns get a gradient.** Tool results are environment text, not model output, and must not
be reinforced. S5 supplies `train_turn_fn(HarnessTurn) -> bool` for this and ships `has_tool_call`;
TRL's white-box path carries the same idea as a `tool_mask`. Our episode is tool-heavy, so the
starting point is: reinforce turns that called a tool, and re-examine whether the final `submit`
turn should be included.

### 3. Trainer choice is left open, deliberately

S5's loop-owning path (`AsyncGRPOTrainer` + `HarnessRolloutWorker`) fits our architecture almost
exactly — its worked example, `opencode`, is a coding agent scored by "the fraction of held-out
tests that passed". But it lives under `trl.experimental` and its API may change, and it is **not**
the path Unsloth's NeMo Gym integration documents (that one is single-turn `GRPOTrainer`). Whether
Unsloth's optimisations compose with `AsyncGRPOTrainer` is **unverified** and is the first thing
the adapter prototype must answer.

## Options considered

1. **Flatten the episode into one completion string.** Rejected. Tool outputs would be trained on
   as though the model produced them unless masked, and the re-tokenised string is not the token
   sequence that was actually sampled — so the logprobs the ratio is computed from are wrong.
2. **One group member per turn.** Rejected. Turns within an episode are not independent samples of
   the same prompt; GRPO's group-relative advantage assumes they are.
3. **Score a failed verify as 0.0** (the upstream default). Rejected — this ADR exists because of it.
4. **Score a failed verify as −1.0.** Rejected. That is the gate constant. A network blip is not a
   reward hack, and conflating them would poison the one signal ADR 0006 spent its coupling
   constraint protecting.
5. **Retry, then score.** Retries are fine and orthogonal; what must not happen is a *number* at
   the end of exhausted retries.

## Consequences

- A rollout is scored, or it is dropped. There is no third outcome and no fabricated score.
- Group sizes become variable. `num_generations: 8` with one discard is a group of 7 — acceptable,
  and far better than a group of 8 containing a counterfeit.
- The discard rate is a first-class training metric, logged every step.
- The adapter is real engineering, ahead of any hyperparameter work.
- Two upstream behaviours are now knowingly diverged from, both recorded in the training config.
  S3 additionally emits a placeholder turn with `generation_token_ids=[0]` when a rollout produced
  no trainable tokens, logging a warning and continuing; that is the same hazard family as the
  `0.0` default and our adapter should discard instead.
