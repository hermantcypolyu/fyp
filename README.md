# rl-coding-agents

**RL of Small Coding Agents with Verifiable Rewards** — benchmark, NeMo Gym environment,
reward implementation and literature base.

> PolyU BSc (Hons) AI & Information Engineering — FYP 2026/27 · `FYP_enivan_20260513235958`
> Student: Yeung Tsz Chun Herman · Supervisor: LAU Chun-Fai · Industry mentor: NVIDIA AI Technology Center, Hong Kong

---

## Status: controlled optimizer update verified; evaluation remains open

On 2026-10-18, one controlled, one-step GRPO-style LoRA optimizer update was completed using
eight rollouts for a single training task. The update produced finite loss and gradients, changed
LoRA parameters, and its checkpoint reloaded exactly. This verifies optimizer and checkpoint
plumbing only; it does **not** show improved coding performance. Full multi-step or large-scale RL,
a controlled before-and-after evaluation, and held-out evaluation remain incomplete. See
[docs/PHASE_F4_CONTROLLED_UPDATE.md](docs/PHASE_F4_CONTROLLED_UPDATE.md) for the evidence and
[docs/TRAINING_READINESS.md](docs/TRAINING_READINESS.md) for the historical handoff and remaining
work.

---

## Quick start (~5 minutes, CPU only)

```bash
git clone <this-repo> && cd rl-coding-agents
uv sync --extra dev          # 1. working environment
make validate                # 2. all 36 tasks pass schema checks
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
| `benchmark/` | 36 project-specific Python repair tasks (26 train / 10 held-out; authored, textbook-pattern, and real-repo provenance) + schema + validator |
| `envs/coding_env/` | NeMo Gym environment: seven-tool resources server, sandbox, verifier, agent harness |
| `rewards/` | `reward_v1` = normalised hidden-test score, plus six hard gates |
| `analysis/` | pass@k with bootstrap CIs, failure taxonomy, rollout trace schema |
| `configs/` | model / eval / training configs; the controlled Phase F4 optimizer smoke is documented separately |
| `docs/` | literature review, reward design, task authoring guide, ADRs, training handoff |
| `scripts/` | serve a model, smoke one task, verify gold patches |
| `results/` | smoke-test artefacts and local, Git-ignored Phase F4 pilot evidence |

---

## Where to go next

| You want to… | Read |
|---|---|
| Present the project to a new audience | [`docs/demo/cantonese_presentation_script.md`](docs/demo/cantonese_presentation_script.md) |
| Walk through the implementation | [`docs/demo/code_walkthrough.md`](docs/demo/code_walkthrough.md) |
| Start RL training | [`docs/TRAINING_READINESS.md`](docs/TRAINING_READINESS.md) ⭐ |
| Understand the reward | [`docs/reward_design.md`](docs/reward_design.md) |
| Add task #41 | [`docs/task_authoring_guide.md`](docs/task_authoring_guide.md) |
| Understand the NeMo Gym integration | [`docs/nemo_gym_notes.md`](docs/nemo_gym_notes.md) |
| See the academic grounding | [`docs/literature_review.md`](docs/literature_review.md) |
| Know why something is the way it is | [`docs/decisions/`](docs/decisions/) |
| See the build history | [`PROGRESS.md`](PROGRESS.md) |

---

## Hard boundaries (see `AGENT_BUILD_SPEC.md` §0.2 and §9)

- No full or multi-step RL run has been completed. One controlled, one-step GRPO-style LoRA
  optimizer update is documented in
  [docs/PHASE_F4_CONTROLLED_UPDATE.md](docs/PHASE_F4_CONTROLLED_UPDATE.md).
- The held-out split is **sealed**; do not open it.
- Hidden tests are never mounted into the agent's sandbox.
- No number appears in any document unless a command produced it and the output is under `results/`.

## Reproducibility & Handoff

This project is organized to be reproducible and portable so you (or another developer) can continue development on another machine.

- **Python version:** 3.12+ is required. Use the same Python version on other machines for best compatibility.
- **Project setup (minimal steps):**

```bash
# clone the repo
git clone <this-repo-root> && cd rl-coding-agents

# create an isolated environment (recommended):
# Option A: using `uv` (preferred if you use `uv` workflows)
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev

# Option B: manually with Python
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if present; otherwise use pyproject tooling
```

- **Environment variables and API keys:** copy `env.example.yaml` to `env.yaml` and fill in provider keys (do NOT commit `env.yaml`).
- **Quick validation:**

```bash
make validate
make test
make smoke    # runs a lightweight end-to-end smoke test
```

- **Reproducible development notes:**
	- Keep the Python runtime at 3.12+. Use the same `uv` and `uv.lock` versions when sharing environments.
	- Do not commit secrets, large model weights, or datasets. Use `env.example.yaml` as a template and a private `env.yaml` for machine-specific secrets.
	- Use branches for feature work and open PRs for mainline changes.

If you want, follow the `HANDOFF.md` file for a concise checklist to move development to another computer.
