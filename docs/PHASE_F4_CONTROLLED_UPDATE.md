# Phase F4 Controlled GRPO-Style LoRA Update

## Status and scope

Completed on 2026-10-18: one optimizer update using a group of eight rollouts for the
`bugfix_filesize_unit_rollover` task. This is an optimizer and checkpoint plumbing smoke test,
not evidence of coding-task improvement.

The base model was Qwen/Qwen2.5-Coder-1.5B-Instruct. The update started from the
`results/raw/rl_smoke_1step/checkpoint` LoRA adapter and wrote the updated adapter to
`results/raw/phase_f4_controlled_pilot_v1/checkpoint`.

## Rollout and group statistics

- Rollout source: `results/raw/b6_rl_step2_filesize_group_8/rollouts.jsonl`
- Rollouts: 8
- Trainable turns: 48
- Trainable tokens: 1,390
- Rewards: `[0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- Reward mean: -0.125
- Reward standard deviation: 0.33071891388307384
- Advantage mean: 0.0
- Advantage standard deviation: 0.9999969762933588

The lowest-reward rollout received the most-negative group-relative advantage.

## Optimizer evidence

- Policy loss: -0.1843777285539545 (finite)
- Raw gradient norm: 0.34685107521512104 (finite)
- Clipped gradient norm: 0.34685108065605164
- Optimizer step completed: yes
- Changed LoRA parameter tensors: 392
- Parameter delta L2: 0.038865488736174576
- Maximum absolute parameter delta: 9.999992471421137e-06

## Checkpoint and runtime evidence

- Checkpoint saved and reloaded: yes
- Reloaded LoRA parameter tensors: 392
- Maximum parameter difference after reload: 0.0
- GPU: NVIDIA RTX A6000
- Peak VRAM: 19,908.5263671875 MiB
- Training time: 23.234683330054395 seconds
- Training exit code: 0

## Evidence locations

Machine-readable evidence and the checkpoint are stored locally under
`results/raw/phase_f4_controlled_pilot_v1/` and are intentionally excluded from normal Git
history. Execution logs and the consolidated archive are stored under `~/fyp_run_logs/`:

- `phase-f4-controlled-pilot-v1.log`
- `phase-f4-controlled-pilot-v1.status`
- `phase-f4-controlled-pilot-v1-provenance.txt`
- `fyp-phase-f4-artifacts-2026-10-18.tar.gz`

Archive SHA-256:

`64194612afde29579f44a8ffd968ad932a269a33931d70acadec94f1ea2f595c`

## Interpretation and limitations

This run verifies that a non-zero-variance reward group produced finite group-relative
advantages, a finite non-zero gradient, a completed optimizer step, changed LoRA parameters,
and a checkpoint that reloads exactly.

It does **not** establish improved coding performance, improved held-out performance, a higher
SWE-Gym Lite solve rate, stability of multi-step online RL, or convergence of longer GRPO
training. Those require controlled before-and-after evaluation on the same tasks, prompts, rollout
count, temperature, tool limits, and reward implementation, followed by a separately controlled
held-out evaluation.

## Next stage

Compare the pre-update adapter at `results/raw/rl_smoke_1step/checkpoint` with the updated
adapter at `results/raw/phase_f4_controlled_pilot_v1/checkpoint` under a fixed evaluation
protocol. Keep the held-out split sealed until the evaluation protocol and checkpoint selection
are finalized.
