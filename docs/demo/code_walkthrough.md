# Pipeline Demo — Code Walkthrough

## Repository map

```text
benchmark/
  tasks/
    train/<task_id>/
      task.yaml       # task instructions, constraints, test metadata
      repo/           # repository the agent works on
      smoke/          # visible tests the agent may run
      tests/          # private grading tests; do not show their contents
      solution/       # reference fix; do not show for held-out tasks
    heldout/<task_id>/
                    # reserved for independent evaluation

envs/coding_env/
  agent.py            # NeMo Gym agent integration
  session.py          # state for one episode
  tools.py            # seven agent tools
  sandbox.py          # internal benchmark filesystem sandbox
  verifier.py         # private grading and gate checks
  docker_sandbox.py   # separate Docker-backed SWE-Gym Lite pilot path

rewards/compose.py    # reward formula
training/rollout_adapter.py
                      # rollout JSON -> trainable episode sample
scripts/run_optimizer_smoke.py
                      # one-step GRPO-style optimizer plumbing test
docs/PHASE_F4_CONTROLLED_UPDATE.md
                      # measured evidence for one internal training task
```

### Current facts to keep straight

- This is a project-specific internal benchmark, not the SWE-Gym Lite dataset.
- The current checkout contains 36 task specs: 26 train and 10 held-out.
- Provenance counts are 4 `authored`, 15 `textbook_pattern`, and 17 `real_repo`.
- `Sandbox` in `sandbox.py` is the internal benchmark's filesystem sandbox. The separate
  `docker_sandbox.py` path is for a Docker-backed SWE-Gym Lite pilot; its presence does not
  mean current internal benchmark runs use Docker or SWE-Gym Lite.
- Phase F4 ran one optimizer update on `bugfix_filesize_unit_rollover`. Its metrics are not
  aggregate benchmark performance.
- Do not reveal the contents of hidden tests or held-out reference solutions.

## 1. Task and dataset

**Show:** `benchmark/tasks/train/feature_interval_merge/task.yaml`, then its `repo/`, `smoke/`,
`tests/`, and `solution/` names in Explorer. Do not open held-out tests or held-out solutions.

**Explain:** Each task has a project-specific task description and a small repository. The
suite combines tasks written for this project, tasks based on common programming patterns,
and tasks derived from real repositories. It follows a SWE-style executable task design, but
it is not the SWE-Gym Lite dataset or schema.

**Task layout:**

```text
benchmark/tasks/train/<task_id>/
├── task.yaml
├── repo/
├── smoke/
├── tests/
└── solution/
```

## 2. Agent harness

**File:** [`envs/coding_env/agent.py`](../../envs/coding_env/agent.py), `CodingAgent.responses()`
**Location:** approximately lines 52–74

```python
class CodingAgent(SimpleAgent):
    config: CodingAgentConfig

    async def responses(
        self,
        request: Request,
        response: Response,
        body: NeMoGymResponseCreateParamsNonStreaming = Body(),
    ) -> NeMoGymResponse:
        cap = _max_turns(body) or self.config.max_steps
        scoped = self.model_copy(
            update={
                "config": self.config.model_copy(
                    update={"max_steps": cap}
                )
            }
        )
        ...
        return await SimpleAgent.responses(
            scoped, request, response, model_body
        )
```

**Brief explanation:** `CodingAgent` reuses NeMo Gym's `SimpleAgent`. This project-specific
wrapper applies the task's turn limit; the inherited agent handles the multi-turn model/tool
interaction.

**Say:** “The harness is the coordinator. It passes context to the model, routes tool calls,
and feeds tool results back for the next turn.”

## 3. Tools

**File:** [`envs/coding_env/tools.py`](../../envs/coding_env/tools.py), `Toolbox.call()`
**Location:** approximately lines 174–194

```python
def call(
    self,
    name: str,
    arguments: dict[str, Any] | None = None,
) -> ToolResult:
    arguments = dict(arguments or {})
    method = getattr(self, name, None)
    if name not in _DISPATCHABLE or method is None:
        result = _err(f"unknown tool {name!r}.")
    elif self.n_calls >= self.max_tool_calls:
        result = _err("tool call budget is exhausted.")
    else:
        result = method(**arguments)
    self.log.append(
        ToolCall(
            name=name,
            arguments=arguments,
            ok=result.ok,
            output=result.output,
        )
    )
    return result
```

The actual implementation also handles malformed tool arguments. Its seven tools are
`list_files`, `read_file`, `write_file`, `edit_file`, `replace_lines`, `run_tests`, and
`submit`.

**Brief explanation:** The model requests an action; the toolbox checks and runs it, then
returns a result. The model does not directly operate on the host filesystem.

**Show the difference between visible tests and submission:**

- `tools.py`, `Toolbox.run_tests()`, approximately lines 544–570: runs visible smoke tests
  only.
- `tools.py`, `Toolbox.submit()`, approximately lines 573–575: ends the episode. Hidden
  verification happens later.

## 4. Episode and sandbox

**File:** [`envs/coding_env/session.py`](../../envs/coding_env/session.py), `EpisodeSession`
**Location:** approximately lines 100–144

```python
def tool(self, name, arguments=None):
    result = self.sandbox.toolbox.call(name, arguments)
    return {
        "ok": result.ok,
        "output": result.output,
        "done": self.done,
    }

def grade(self) -> VerifyResult:
    if self._result is None:
        self._result = verify_sandbox(
            self.sandbox,
            limits=self.limits,
        )
    return self._result
```

**File:** [`envs/coding_env/sandbox.py`](../../envs/coding_env/sandbox.py), `Sandbox.start()`
**Location:** approximately lines 57–72

```python
root = base / "work"
materialise(
    task,
    root,
    include_tests=False,
    include_smoke=True,
)
if (root / "tests").exists():
    raise RuntimeError("hidden tests leaked into the sandbox")
```

**Brief explanation:** Each episode gets its own working copy. The internal benchmark
sandbox materializes the repository and visible smoke tests, but not the private grading
suite.

**Important distinction:** [`envs/coding_env/sandbox.py`](../../envs/coding_env/sandbox.py)
is the current internal benchmark filesystem sandbox.
[`envs/coding_env/docker_sandbox.py`](../../envs/coding_env/docker_sandbox.py) is a separate
Docker-backed SWE-Gym Lite pilot path. Docker is an execution technology, not a task dataset.

## 5. Verifier and reward

**File:** [`envs/coding_env/verifier.py`](../../envs/coding_env/verifier.py), `verify()` and `_build_grading_copy()`
**Locations:** approximately lines 148–220 and 222–228

`verify()` builds a fresh grading copy, runs the task's private tests, checks gates, and
returns a structured result. The grading copy combines pristine task tests with the agent's
modified repository.

**File:** [`rewards/compose.py`](../../rewards/compose.py), `normalised_reward()` and `compose_reward()`
**Locations:** approximately lines 58–84

```text
raw_reward = (passed - floor_tests)
             ---------------------
             (n_tests - floor_tests)

reward = clamp(raw_reward, -1, 1)
```

- `passed`: hidden tests passed by the submitted version.
- `n_tests`: declared total hidden tests.
- `floor_tests`: tests already passed by the untouched repository.
- A hard gate sets the reward to `-1`.

For an easy arithmetic example, if there are 10 tests, the untouched repository passes 4,
and the agent's version passes 7:

```text
(7 - 4) / (10 - 4) = 0.5
```

That means the agent passed half of the tests it could improve. This is an illustrative
example, not a reported run result.

**Say:** “The agent can receive a score, but it does not get to inspect the hidden test
cases.”

## 6. Rollout adapter

**File:** [`training/rollout_adapter.py`](../../training/rollout_adapter.py), `adapt_rollout()`
**Location:** approximately lines 96–195

```python
generation_ids = item.get("generation_token_ids")

# Tool results without model-generated tokens are context only.
if not generation_ids:
    continue

prompt_ids = _require_int_tokens(
    item.get("prompt_token_ids"),
    field="prompt_token_ids",
    output_index=output_index,
)
generation_ids_tuple = _require_int_tokens(
    generation_ids,
    field="generation_token_ids",
    output_index=output_index,
)
generation_log_probs = _require_log_probs(
    item.get("generation_log_probs"),
    output_index=output_index,
)
```

**Brief explanation:** An interaction trace contains both model actions and tool replies.
The adapter retains turns with captured model-generated tokens and validates their token IDs
and old log probabilities. Tool output is context, not a model action to train.

## 7. Group-relative advantage

**File:** [`scripts/run_optimizer_smoke.py`](../../scripts/run_optimizer_smoke.py), `compute_advantages()`
**Location:** approximately lines 117–143

```python
reward_mean = statistics.fmean(rewards)
reward_std = statistics.pstdev(rewards)

if reward_std <= 0:
    raise RuntimeError("Reward standard deviation is zero.")

advantages = [
    (reward - reward_mean) / (reward_std + 1.0e-6)
    for reward in rewards
]
```

The normalized group-relative advantage is:

$$
A_i = \frac{R_i-\mu_R}{\sigma_R+10^{-6}},
\qquad
\mu_R=\frac{1}{G}\sum_{j=1}^{G}R_j,
\qquad
\sigma_R=\sqrt{\frac{1}{G}\sum_{j=1}^{G}(R_j-\mu_R)^2}
$$

Here, \(R_i\) is rollout \(i\)'s reward, \(G\) is the number of rollouts in the group,
\(\mu_R\) is the group mean, and \(\sigma_R\) is the population standard deviation
(`statistics.pstdev`). A positive advantage means it scored above its group average; a
negative advantage means it scored below. The script requires one task per group and stops
if every reward is identical, because there is no relative signal.

## 8. Policy loss and optimizer update

**File:** [`scripts/run_optimizer_smoke.py`](../../scripts/run_optimizer_smoke.py)
**Locations:** ratio/objective approximately lines 456–485; optimizer step approximately
lines 524–559.

```python
log_ratio = current_log_probs - old_log_probs
ratio = torch.exp(
    torch.clamp(log_ratio, min=-20.0, max=20.0)
)
clipped_ratio = torch.clamp(
    ratio,
    1.0 - args.clip_epsilon,
    1.0 + args.clip_epsilon,
)

unclipped_objective = ratio * advantage_tensor
clipped_objective = clipped_ratio * advantage_tensor
surrogate = torch.minimum(
    unclipped_objective,
    clipped_objective,
)
turn_loss = -surrogate.sum() / total_trainable_tokens
turn_loss.backward()
```

The code implements the following token-weighted objective:

$$
\mathcal{L}(\theta)
= -\frac{1}{N_{\text{train}}}
\sum_{i=1}^{G}\sum_{t\in\mathcal{T}_i}
\min\left(
r_{i,t}(\theta)A_i,\;
\operatorname{clip}\left(r_{i,t}(\theta),1-\epsilon,1+\epsilon\right)A_i
\right)
$$

Here, \(G\) is the number of rollouts, and \(\mathcal{T}_i\) is the set of trainable
generated tokens in rollout \(i\). The total number of trainable tokens is:

$$
N_{\text{train}}=\sum_{i=1}^{G}\left|\mathcal{T}_i\right|
$$

The probability ratio for token \(t\) in rollout \(i\) is:

$$
r_{i,t}(\theta)
=\exp\!\left(
\operatorname{clamp}\!\left(
\ell_{i,t}(\theta)-\ell_{i,t}^{\text{old}},\,-20,\,20
\right)
\right)
$$

Here, \(\ell_{i,t}(\theta)\) is the updated policy's log probability for the token, and
\(\ell_{i,t}^{\text{old}}\) is the log probability under the policy that generated the
rollout. The numerical clamp prevents overflow; it is separate from the PPO clipping to
\(1-\epsilon\) and \(1+\epsilon\) in the objective above.

- The ratio compares the updated policy with the policy that generated the rollout.
- Clipping limits how much the ratio can influence one update.
- The loss uses only trainable generated tokens.
- `backward()` computes gradients; `optimizer.step()` changes the LoRA parameters.

**Formula boundary:** The illustration in the presentation includes a KL term and a value
model/critic, but this optimizer smoke script implements neither. Do not add those terms to
the formula when describing this run.

## 9. Save, reload, and evidence

**File:** [`scripts/run_optimizer_smoke.py`](../../scripts/run_optimizer_smoke.py)
**Locations:** save around lines 557–559; reload and exact comparison around lines 631–674.

The script saves the LoRA adapter, reloads it, and checks that the reloaded parameters match.
The documented Phase F4 result is from `bugfix_filesize_unit_rollover`, not every task in the
benchmark:

- 8 rollouts, 48 trainable turns, and 1,390 trainable tokens.
- 392 LoRA parameter tensors changed.
- Reload maximum parameter difference: 0.0.

**Correct conclusion:** The one-step optimizer/checkpoint plumbing worked. This does not
prove the model's coding performance improved.

## 10. Future task source: SWE-Gym Lite

**File:** [`envs/coding_env/docker_sandbox.py`](../../envs/coding_env/docker_sandbox.py)

This is a separate Docker-backed pilot path. Future integration requires mapping task
metadata, repository/base commit, runtime dependencies, tests, and train/evaluation splits.
Do not say that the current internal benchmark is SWE-Gym Lite.

## Short presentation route

If time is limited, show only:

1. `benchmark/tasks/train/` — what one task contains.
2. `tools.py` — how an agent uses tools.
3. `verifier.py` and `rewards/compose.py` — how a submission is scored.
4. `rollout_adapter.py` — which turns become training data.
5. `run_optimizer_smoke.py` and the Phase F4 report — what one update proved.
