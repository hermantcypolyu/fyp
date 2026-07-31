# rl-coding-agents

**RL of Small Coding Agents with Verifiable Rewards** — benchmark, NeMo Gym environment,
reward implementation and literature base.

> PolyU BSc (Hons) AI & Information Engineering — FYP 2026/27 · `FYP_enivan_20260513235958`
> Student: Yeung Tsz Chun Herman · Supervisor: LAU Chun-Fai · Industry mentor: NVIDIA AI Technology Center, Hong Kong

---

## ⛔ Status: BUILD-ONLY PHASE — no training has been run

This repository is a **cold-start-ready** project folder. Everything needed to begin RL training
exists, is tested and is documented. **No RL training, no GRPO, no fine-tuning, and no full
evaluation sweep has been executed.** See [docs/TRAINING_READINESS.md](docs/TRAINING_READINESS.md)
for exactly what exists, what was never run, and the precise next command.

---

## Quick start (~5 minutes, CPU only)

```bash
git clone <this-repo> && cd rl-coding-agents
uv sync --extra dev          # 1. working environment
make validate                # 2. all 40 tasks pass schema checks
make verify-gold             # 3. every gold patch fixes its task; untouched repo fails
make test                    # 4. reward gates, metrics, sandbox isolation
make smoke                   # 5. one task end-to-end -> a real reward
```

Windows without GNU make: use `./make.ps1 validate`, `./make.ps1 smoke`, etc.

Run `make help` for every target.

---

## What is here

| Path | What it is |
|---|---|
| `benchmark/` | 40 hand-authored Python repair tasks (30 train / 10 held-out) + schema + validator |
| `envs/coding_env/` | NeMo Gym environment: 6-tool resources server, sandbox, verifier, agent harness |
| `rewards/` | `reward_v1` = hidden-test pass fraction, plus four hard-zero gates |
| `analysis/` | pass@k with bootstrap CIs, failure taxonomy, rollout trace schema |
| `configs/` | model / eval / training configs — **training configs are written but never executed** |
| `docs/` | literature review, reward design, task authoring guide, ADRs, training handoff |
| `scripts/` | serve a model, smoke one task, verify gold patches |
| `results/` | smoke-test artefacts only in this phase |

---

## Where to go next

| You want to… | Read |
|---|---|
| Start RL training | [`docs/TRAINING_READINESS.md`](docs/TRAINING_READINESS.md) ⭐ |
| Understand the reward | [`docs/reward_design.md`](docs/reward_design.md) |
| Add task #41 | [`docs/task_authoring_guide.md`](docs/task_authoring_guide.md) |
| Understand the NeMo Gym integration | [`docs/nemo_gym_notes.md`](docs/nemo_gym_notes.md) |
| See the academic grounding | [`docs/literature_review.md`](docs/literature_review.md) |
| Know why something is the way it is | [`docs/decisions/`](docs/decisions/) |
| See the build history | [`PROGRESS.md`](PROGRESS.md) |

---

## Hard boundaries (see `AGENT_BUILD_SPEC.md` §0.2 and §9)

- No GRPO / PPO / DPO / SFT runs in this phase.
- The held-out split is **sealed**; do not open it.
- Hidden tests are never mounted into the agent's sandbox.
- No number appears in any document unless a command produced it and the output is under `results/`.
