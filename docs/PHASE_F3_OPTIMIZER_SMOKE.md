# Phase F3 Optimizer-Step Smoke Verification

## Status

Phase F3 is complete.

Two historical one-step GRPO-style LoRA optimizer smoke runs were
re-audited and verified.

## Model and adapter

- Base model: Qwen/Qwen2.5-Coder-1.5B-Instruct
- Adapter type: LoRA
- Rank: 16
- Alpha: 32
- Dropout: 0.0
- Learning rate: 1e-5
- Target modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

## Run 1

- Directory: results/raw/rl_smoke_1step
- Task: bugfix_dotenv_bom_prefix
- Policy loss: -0.17073625142802484
- Raw gradient norm: 0.307842468207567
- Changed LoRA parameters: 196
- Parameter delta L2: 0.032080758779330344
- Reload maximum parameter difference: 0.0
- Adapter SHA-256:
  689af5b9e05815fd4e6bcea3e059389e15bdb51f64f5dea78eff6c7455f89150

## Run 2

- Directory: results/raw/rl_smoke_step2
- Task: bugfix_filesize_unit_rollover
- Policy loss: -0.1843777285539545
- Raw gradient norm: 0.3466873121306448
- Changed LoRA parameters: 392
- Parameter delta L2: 0.03886459382075283
- Reload maximum parameter difference: 0.0
- Adapter SHA-256:
  b406969ea77d0890b2560668f1c1416baad25066f4dce6f1281efb3569c6b02e

## Tensor-level comparison

- Adapter tensor count: 392
- Changed tensors: 392
- Unchanged tensors: 0
- Total values: 18,464,768
- Changed values: 18,464,735
- Total absolute delta: 163.09988828003407
- Maximum absolute delta: 9.999992471421137e-06

## Verified properties

- Rewards had non-zero group variance.
- Group-relative advantages were finite.
- The lowest-reward rollout received the most-negative advantage.
- Model action tokens were included in policy loss.
- Prompt and environment-observation tokens were excluded.
- Policy losses were finite.
- Gradient norms were finite and non-zero.
- Optimizer steps completed.
- LoRA parameters changed.
- Checkpoints were saved.
- Checkpoints were reloaded.
- Reloaded parameters matched saved parameters exactly.

## Interpretation

These results verify the optimizer and checkpoint plumbing.

They do not demonstrate improved coding performance. A controlled
before-and-after RL pilot and held-out evaluation are still required.
