# 0005 — Build our own environment rather than reuse the NeMo Gym SWE agents

- **Status:** Accepted
- **Date:** 2026-07-31
- **Milestone:** B0
- **Required by:** `AGENT_BUILD_SPEC.md` §6.2 ("survey NeMo Gym's environment hub for an
  existing SWE-style environment; if one exists, say why we are or are not reusing it")

## Context

Before writing an environment, we surveyed the hub at commit `ec5a8136`:
**101 resources servers, 33 agents** (`gym list resources-servers`, `gym list agents`).

Relevant prior art:

| Item | What it is | Why it does not close our gap |
| --- | --- | --- |
| `mini_swe_agent`, `mini_swe_agent_2`, `swe_agents`, `anyswe_agent` | agent harnesses for SWE-bench-style repo tasks | Harnesses, **not benchmarks**. They assume an existing task corpus and a shell-based action space. |
| `code_gen`, `bigcodebench`, `evalplus` | function-level code generation | Single file, single turn, no repo, no editing. Wrong task shape. |
| `competitive_coding_challenges`, `code_fim` | contest problems / fill-in-the-middle | Same — no multi-turn repository interaction. |
| `cvdp` | NVIDIA's code/verification env | Hardware-verification oriented; different domain. |

So: **a suitable agent harness exists; a suitable task corpus does not.** Nothing in the hub
provides small, self-contained, gold-patched repositories with *hidden* tests sized for a 1.5B
model on 12 GB.

Two further constraints push the same way:

- **SWE-bench is the wrong difficulty.** Its instances are large real-world repos where strong
  frontier models score modestly. A 1.5B model would score ≈0 almost everywhere, giving **zero
  reward variance** — and GRPO's advantage is computed *within* a group of samples for the same
  prompt. All-zero groups produce zero gradient. A benchmark a small model always fails is not
  merely discouraging; it is mathematically useless for this algorithm.
- **A shell tool is a huge action space.** `mini_swe_agent` gives the model `bash`. For a 1.5B
  model that is mostly a way to generate invalid commands, and it makes the sandbox boundary much
  harder to prove.

## Options considered

1. **Adopt `swe_agents` + SWE-bench wholesale.** Rejected: no reward variance at 1.5B (above),
   needs Docker per instance, and the corpus is far larger than a 12 GB single-GPU budget allows
   to iterate on.
2. **Adopt a `*swe_agent*` harness, author our own tasks for it.** Tempting, and we *do* borrow
   ideas. Rejected as the base because the shell action space conflicts with §5.5's fixed 6-tool
   surface, and because their sandboxing assumes container runtimes we cannot rely on.
3. **Author our own 40-task benchmark; build a resources server on
   `SimpleResourcesServer`; subclass `simple_agent`.** ✅ Chosen.

## Decision

Build `envs/coding_env/` from the two smallest, most general built-ins:

- **server** ← `example_session_state_mgmt` — its `session_id → state` map becomes
  `session_id → sandbox directory`, which is our whole isolation story.
- **agent** ← `simple_agent` — ~200 lines, domain-agnostic tool-call loop. Our subclass adds only
  the `max_turns` / `max_tool_calls` accounting §5.5 requires.
- **reward** ← `example_tool_call_multireward`'s `reward_components`, so `reward_v1` ships next to
  its gate breakdown and the B4 failure taxonomy is computable without re-running anything.

The action space stays the **fixed 6 tools of §5.5 — no shell**. This is a real design choice, not
an omission, and it is defended in `docs/literature_review.md`: a small model benefits from a
small, well-typed action space; every tool is individually verifiable; and "no shell" makes the
sandbox boundary provable by construction rather than by policy (see the isolation test in B3).

Difficulty is calibrated so gold patches are 1–20 lines in repos under ~500 LOC, deliberately
targeting a **non-zero, non-saturated** pass rate for a 1.5B model — the regime where GRPO
actually has signal.

## Consequences

- ✅ Reward variance exists at 1.5B, which is the precondition for GRPO to learn anything.
- ✅ Tasks run in seconds on CPU, so `make validate` and `make smoke` stay fast and container-free.
- ✅ Partial credit is real (§5.4: several small assertions per task), giving a dense reward
  instead of a sparse 0/1.
- ❌ **Results will not be comparable to published SWE-bench numbers.** We must never imply they
  are. Every reported number must be labelled as being on *our* benchmark.
- ⚠️ Authoring 40 tasks is the single largest chunk of manual work in B2, and benchmark quality
  now bounds the whole project. Mitigated by §5.3's ≥50% `real_repo` provenance rule, the
  gold-patch/untouched-repo double check in `scripts/verify_gold_patches.py`, and sealing the
  held-out split by commit SHA.
- ⚠️ If the tasks turn out too easy (pass rate saturates near 1.0), reward variance dies from the
  other end. `docs/TRAINING_READINESS.md` §6 lists this as a known unknown, measurable by a
  baseline eval **before** any training.
