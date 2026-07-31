# AGENT_BUILD_SPEC.md — v2 (Build-Only)
## RL Coding Agents · NeMo Gym Benchmark + Environment + Literature Base

> **Read this file completely before writing any code. Re-read §0 at the start of every session.**
> If this document and your own judgement disagree, this document wins.
> If this document is silent, **stop and ask the human** — do not invent scope.

| Field | Value |
|---|---|
| Project | RL of Small Coding Agents with Verifiable Rewards |
| Programme | PolyU BSc (Hons) AI & Information Engineering — FYP 2026/27 |
| Project ID | FYP_enivan_20260513235958 |
| Student | Yeung Tsz Chun Herman |
| Supervisor | LAU Chun-Fai |
| Industry mentor | NVIDIA AI Technology Center, Hong Kong |
| Local hardware | **RTX 4070 Ti — 12 GB GDDR6X** |
| Target model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` |
| Repo name | `rl-coding-agents` |

---

# 0. MISSION & HARD BOUNDARY

## 0.1 What you are building
A **complete, cold-start-ready project folder**. The human will open it later and begin RL training
immediately, with zero setup archaeology. Everything must already be in place, tested, and documented.

## 0.2 What you are NOT doing — ABSOLUTE BOUNDARY 🚫

**DO NOT RUN ANY RL TRAINING. DO NOT LAUNCH GRPO. DO NOT FINE-TUNE ANY MODEL.**

Also forbidden in this phase:
- ❌ No GRPO / PPO / DPO / SFT runs of any kind
- ❌ No full baseline evaluation sweeps (a ≤3-task smoke test is the maximum)
- ❌ No multi-hour GPU jobs
- ❌ No renting cloud GPUs, no spending money
- ❌ No downloading model weights >4 GB without asking

If you believe a milestone requires training to complete, **you have misread the milestone. Stop and ask.**

## 0.3 The deliverable test
The build succeeds if a person can clone the repo on a fresh machine and, within ~30 minutes:
1. `uv sync` → working environment
2. Read `README.md` → understand what exists and what to do next
3. Run `make validate` → all 40 tasks pass schema + gold-patch checks
4. Run `make smoke` → one task runs end-to-end through the Gym environment and returns a reward
5. Read `docs/TRAINING_READINESS.md` → know the exact next command to start RL, and on what hardware

If any of those five fails, the build is not done.

---

# 1. Agent Operating Rules

### 1.1 One milestone at a time
Milestones **B0 → B6** (§4). Finish one, tick every Definition-of-Done box, append to `PROGRESS.md`,
**then stop for human review**. Never run two milestones together.

### 1.2 Never fabricate — code, numbers, or citations
- No number appears anywhere unless a real command produced it and the output is saved under `results/`.
- **No academic citation is written unless you have actually retrieved the paper and confirmed title,
  authors, venue and year.** Hallucinated references are the single fastest way to destroy academic
  credibility. If you cannot verify a paper, write `[UNVERIFIED — needs human check]` next to it.
- No API is used from memory — see §1.3.

### 1.3 Ground yourself in real source code
NeMo Gym and NeMo RL move fast. **Never write code against remembered APIs.**
Before implementing anything that touches them: clone the repo, open the closest built-in example,
read it, and mirror its structure. Put the source path you copied from in a code comment:
```python
# Pattern adapted from: nemo-gym/<actual/path/to/example.py> @ commit <sha>
```

Reference documentation:
- NeMo Gym overview — https://docs.nvidia.com/nemo/gym/about/
- Environment anatomy — https://docs.nvidia.com/nemo/gym/about/concepts/environments/
- NeMo RL GRPO tutorial — https://docs.nvidia.com/nemo/gym/tutorials/training-tutorials/nemo-rl-grpo
- Training tutorials index — https://docs.nvidia.com/nemo/gym/tutorials/training-tutorials
- Model card — https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B

### 1.4 Local hardware discipline (12 GB)
- vLLM serving of a 1.5B model is fine. Anything bigger → ask first.
- Cap any smoke test at **3 tasks × 1 sample**. Never a full sweep.
- If you hit CUDA OOM, **do not silently shrink settings and retry** — log it in
  `docs/vram_notes.md` and continue. That file becomes real evidence for the training-phase config.

### 1.5 Small commits, clean repo
Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`, `lit:`). One logical change each.
Never commit: weights, checkpoints, `.venv/`, caches, secrets, blobs >10 MB.
Secrets live in `.env` (gitignored); `.env.example` documents keys with dummy values.

### 1.6 Ask, don't assume
Stop and ask when: a DoD is ambiguous · you want to change the task schema or reward · a dependency
needs a Python/CUDA downgrade · you think you need to train something · you want to add a top-level
directory · a task you're authoring feels unsolvable or trivial.

---

# 2. Hardware Reality (write this down first)

Local card is a **RTX 4070 Ti, 12 GB GDDR6X, 192-bit**. The official NeMo RL GRPO tutorial expects
**1+ nodes of 8× 80 GB GPUs (H100/A100), 64 GB+ RAM, 100 GB+ shared storage**. We are ~53× short of
that. Therefore:

| Work | Runs locally? |
|---|---|
| Benchmark authoring, schema, validation | ✅ CPU only |
| NeMo Gym environment, tools, verifier | ✅ **NeMo Gym needs no GPU at all** |
| Sandbox + pytest execution | ✅ CPU only |
| vLLM serving Qwen 1.5B for smoke tests | ✅ Fits in 12 GB |
| **GRPO training** | ❌ **Not in this phase. Not on this card.** |

The GPU-free property of NeMo Gym is *why this build-only phase is worth doing now* — roughly 70% of
the project's engineering has no compute dependency whatsoever.

**Your job includes making the training phase cheap when it happens.** Every design choice should
shrink future VRAM: small repos, truncated tool outputs, short max sequence length, low turn caps.

---

# 3. Repository Layout

Create exactly this. No new top-level directories without approval.

```
rl-coding-agents/
├── README.md                      # human entry point — what exists, how to run, what's next
├── AGENT_BUILD_SPEC.md            # this file (read-only for you)
├── PROGRESS.md                    # append-only session log (§10)
├── Makefile                       # validate · smoke · catalogue · lint
├── pyproject.toml                 # uv-managed
├── .env.example
├── .gitignore
│
├── docs/
│   ├── TRAINING_READINESS.md      # ⭐ THE HANDOFF DOC — see §7
│   ├── literature_review.md       # ⭐ §6
│   ├── bibliography.bib           # verified BibTeX only
│   ├── task_authoring_guide.md
│   ├── reward_design.md
│   ├── vram_notes.md
│   ├── nemo_gym_notes.md          # what you learned reading the real source
│   └── decisions/                 # ADR: Context → Options → Decision → Consequences
│
├── benchmark/
│   ├── tasks/
│   │   ├── train/<task_id>/       # 30 tasks
│   │   └── heldout/<task_id>/     # 10 tasks — SEALED
│   ├── schema/
│   │   ├── task.schema.json
│   │   └── validate_tasks.py
│   ├── tools/
│   │   ├── new_task.py            # scaffold a blank task
│   │   └── build_catalogue.py
│   └── catalogue.md               # generated
│
├── envs/coding_env/
│   ├── resources_server/          # tools + /verify
│   ├── agent/                     # thin subclass of a built-in harness
│   ├── sandbox/                   # isolation + pytest runner
│   └── configs/
│
├── rewards/
│   ├── components.py              # pure functions, one per term
│   ├── compose.py                 # composition + hard-zero gates
│   └── tests/
│
├── configs/
│   ├── seeds.yaml
│   ├── model/qwen25_coder_1_5b.yaml
│   ├── eval/baseline.yaml
│   └── training/                  # ⚠️ WRITTEN BUT NEVER EXECUTED
│       ├── grpo_smoke.yaml        # 2-step config for the future first run
│       ├── grpo_full.yaml         # NeMo RL, 8×80GB target
│       └── grpo_lowvram.yaml      # QLoRA fallback, documented assumptions
│
├── scripts/
│   ├── serve_model.sh
│   ├── smoke_one_task.py
│   └── verify_gold_patches.py
│
├── analysis/
│   ├── metrics.py                 # pass@k + bootstrap CI (implemented, untested on real runs)
│   ├── failure_taxonomy.py        # classifier skeleton + label enum
│   └── trace_schema.json
│
└── results/                       # only smoke-test artefacts in this phase
    └── raw/<run_id>/
```

---

# 4. Milestones

## B0 — Foundations & real-API grounding *(no GPU)*
- [ ] Repo scaffolded per §3; `uv sync` works from a clean clone; Python 3.12+
- [ ] `Makefile` with `validate`, `smoke`, `catalogue`, `lint` targets (may be stubs initially)
- [ ] `nvidia-smi` output recorded in `docs/vram_notes.md`
- [ ] NeMo Gym cloned locally; **one built-in environment inspected and run unmodified** ← hard gate
- [ ] `docs/nemo_gym_notes.md` written from the **actual source**: how a resources server is registered,
      how tools are declared, what `/verify` receives and returns, how the agent server loops, what the
      dataset/task record looks like, which built-in harness is the best base for us and why
- [ ] `.env.example`, `.gitignore`, `README.md` skeleton committed

## B1 — Task schema & 3 reference tasks *(no GPU)*
- [ ] `task.schema.json` finalised per §5.1
- [ ] `validate_tasks.py` gives clear, actionable errors on malformed tasks
- [ ] `new_task.py` scaffolds a blank task directory
- [ ] **3 fully-worked reference tasks**: one `bug_fix`, one `test_update`, one `config_fix`
- [ ] For each: gold patch makes hidden tests **pass**, untouched repo makes them **fail**
- [ ] `docs/task_authoring_guide.md` complete enough for a human to add task #4 unaided
- [ ] 🛑 **Stop for human review — the schema is a contract; get it approved before mass production**

## B2 — Full benchmark: 30 train + 10 held-out *(no GPU)*
- [ ] 30 train tasks across the 5 categories, difficulty-tiered (§5.2)
- [ ] 10 held-out tasks, authored to the **same distribution**, then **sealed** (§5.4)
- [ ] 100% pass `validate_tasks.py`
- [ ] `verify_gold_patches.py` green for all 40: gold passes, no-op fails
- [ ] Every task records **provenance** (§5.3) — where the bug idea came from
- [ ] `benchmark/catalogue.md` generated: id · category · difficulty · n_tests · LOC · provenance
- [ ] Deliver in **batches of 5–8 tasks** for human review, not one giant drop

## B3 — NeMo Gym environment *(CPU + light GPU)*
- [ ] Resources server exposing the §5.5 tool surface + `POST /verify`
- [ ] Agent harness as a **thin subclass of a built-in** — do not write a loop from scratch (§5.6)
- [ ] Sandbox: agent works on a copy; hidden tests mounted outside its writable root; verifier
      restores pristine tests before grading; no network; CPU/mem caps; hard timeout
- [ ] Rollout trace persisted exactly per `analysis/trace_schema.json`
- [ ] `make smoke` runs **one** task end-to-end and returns a real reward
- [ ] Isolation test proves the agent cannot read or write hidden tests ← hard gate

## B4 — Rewards, metrics & analysis skeleton *(no GPU)*
- [ ] `rewards/components.py`: each term a pure, unit-tested function
- [ ] `rewards/compose.py`: `reward_v1` = hidden-test pass fraction + hard-zero gates (§5.7)
- [ ] Gate tests: a patch touching `tests/` scores exactly `0.0`
- [ ] `analysis/metrics.py`: pass@k + bootstrap 95% CI, unit-tested on synthetic data
- [ ] `analysis/failure_taxonomy.py`: label enum + classifier skeleton (rules may be TODO)
- [ ] `docs/reward_design.md`: v1 rationale + the shaping terms deliberately deferred, with reasons

## B5 — Literature base *(no GPU)*
See §6 for the full brief. Summary DoD:
- [ ] `docs/literature_review.md` — ~10–15 pages, structured per §6.2
- [ ] `docs/bibliography.bib` — **every entry verified**, ≥20 entries
- [ ] Each of our 4 core design choices explicitly justified against the literature (§6.3)
- [ ] Gaps & open questions section — where our study sits, honestly
- [ ] Any unverifiable claim marked `[UNVERIFIED]`

## B6 — Handoff packaging
- [ ] `docs/TRAINING_READINESS.md` complete per §7 ← **the most important file in the repo**
- [ ] `configs/training/*.yaml` written, commented, and clearly marked **NEVER EXECUTED**
- [ ] `README.md` finished: what exists · how to validate · how to smoke · what's next · what's blocked
- [ ] Fresh-clone test: the §0.3 five-step check passes end to end
- [ ] Final `PROGRESS.md` entry summarising the whole build

---

# 5. Build Specifications

### 5.1 `task.yaml` contract
```
benchmark/tasks/{split}/{task_id}/
├── task.yaml
├── repo/              # what the agent sees
├── tests/             # HIDDEN — never mounted into the sandbox
├── smoke/             # optional: visible tests the agent MAY run
└── solution/gold.patch
```

```yaml
id: bugfix_offbyone_pagination      # snake_case, unique, stable forever
category: bug_fix                   # bug_fix|test_update|config_fix|small_feature|docs_improve
difficulty: easy                    # easy|medium|hard
split: train

instruction: >
  The paginate() helper in src/pagination.py returns the wrong slice on the
  final page. Fix it so the last page contains the remaining items.

repo:
  root: repo/
  loc: 180                          # keep small — context budget matters
  entrypoint_hint: src/pagination.py   # omit for harder tiers

verification:
  runner: pytest
  test_files: [tests/test_pagination.py]
  timeout_s: 60
  n_tests: 6                        # must match real collected count — CI enforces

constraints:
  allowed_write_globs: ["repo/src/**", "repo/*.py", "repo/*.cfg", "repo/*.toml"]
  forbidden_write_globs: ["tests/**", "**/conftest.py", "solution/**", "task.yaml"]
  max_turns: 20
  max_tool_calls: 40

provenance:
  source: real_repo                 # real_repo | textbook_pattern | authored
  origin: "github.com/<org>/<repo> @ <commit>"   # required if source=real_repo
  license: MIT
  notes: "Inverted the fix commit; kept the accompanying test."

metadata:
  expected_files_changed: 1
  created: 2026-08-XX
```

### 5.2 Category & difficulty mix

| Category | Train | Held-out | Tests |
|---|---|---|---|
| `bug_fix` | 10 | 4 | read failing test → localise → minimal fix |
| `test_update` | 6 | 2 | update tests after an intentional API change |
| `config_fix` | 5 | 1 | broken `pyproject`/`setup.cfg`/import path |
| `small_feature` | 6 | 2 | add a function to spec |
| `docs_improve` | 3 | 1 | docstrings validated by doctest/signature checks |
| **Total** | **30** | **10** | |

Difficulty per split ≈ **40% easy / 40% medium / 20% hard**.
Easy = one file, obvious fix. Medium = must run tests to localise. Hard = multi-file, or the first
plausible fix breaks something else.

### 5.3 Provenance — prefer real bugs over invented ones ⭐
Model-invented bugs skew toward clean textbook errors (off-by-one, wrong operator, missing `None`
check) because that's what's dense in training data. That biases the whole benchmark.

**Preferred sourcing:** mine small permissively-licensed Python repos for `fix:` commits that also
touched a test. Invert the fix, keep the test. This gives you a real bug distribution, a free gold
patch, a free discriminating test, and a citable provenance line.

**Target: ≥50% of tasks with `source: real_repo`.** Record the origin commit for every one.
Where you do author a task yourself, mark it honestly — do not dress up an invented bug as a real one.

### 5.4 Authoring rules
1. Every task **solvable** by its gold patch — CI proves it.
2. Every task **failable** — untouched repo fails ≥1 test.
3. Tests **deterministic**: no network, no clock, no unseeded randomness, no outside-sandbox I/O.
4. Tests **hidden**: the agent sees only pass/fail counts and a truncated stderr tail.
5. **Partial credit must be real** — several small assertions beat one giant test, so the reward is a
   smooth signal rather than 0/1.
6. **Repos tiny** (< ~500 LOC, ideally < 300). This is a tool-use study, not a long-context study —
   and it directly reduces future training VRAM.
7. Held-out tasks match the train distribution exactly. Otherwise the comparison measures
   distribution shift, not learning.
8. **Sealing:** when held-out is done, record the commit SHA in `PROGRESS.md` and do not open those
   files again.

### 5.5 Tool surface — keep it minimal
Fewer tools = smaller action space = better sample efficiency at 1.5B. Ship exactly:

| Tool | Signature | Notes |
|---|---|---|
| `list_files` | `(path=".")` | sandbox-relative only |
| `read_file` | `(path, start=None, end=None)` | hard truncation; return line numbers |
| `write_file` | `(path, content)` | rejected if it violates `allowed_write_globs` |
| `edit_file` | `(path, old_str, new_str)` | exact match; error if 0 or >1 hit |
| `run_tests` | `()` | runs **visible smoke tests only** — never the hidden suite |
| `submit` | `()` | ends the episode → triggers hidden verification |

Implementation notes:
- Tool errors must be **short and instructive**. A 1.5B model recovers from
  `"No match for old_str in src/foo.py"` far better than from a traceback.
- Truncate every tool output to a strict token budget. Runaway pytest output eats the context window
  and causes failures that have nothing to do with coding ability.
- `run_tests` must never expose the grading signal, or eval/train separation collapses.

### 5.6 Environment structure
NeMo Gym defines an environment as **dataset + agent harness + verifier + state**. Ours:

| Gym component | Ours |
|---|---|
| Dataset | `benchmark/tasks/**` |
| Agent harness | thin subclass of a built-in, with our 6 tools |
| Verifier | sandboxed pytest → structured result → reward |
| State | mutable repo working copy in the sandbox |

The architecture is HTTP microservices: a head server for lifecycle, an **agent server** (`POST /run`),
a **model server** (`POST /v1/chat/completions`), and a **resources server** hosting tools and
`POST /verify`. Nearly all our work lives in the resources server plus a thin agent subclass.

Built-ins worth reading before you choose: `simple_agent` (tool-augmented while loop — likely our
base), `mini_swe_agent` / `swe_agents` (repo mounting, sandboxing, patch extraction), and
`verifiers_agent` (notable because it **preserves token IDs and logprobs**, which matters for RL —
check what NeMo RL needs before committing).

⭐ **Critical invariant:** the *same* verifier produces evaluation metrics and the RL reward.
Implement it once, call it from both paths. If eval and training ever disagree about what "success"
means, the entire study is invalid.

### 5.7 Reward v1 — and only v1
```
reward_v1 = fraction of hidden unit tests passing after the final patch   ∈ [0,1]
```
**Hard-zero gates** (not penalties — they zero the reward outright):

| Gate | Trigger |
|---|---|
| `tests_modified` | wrote to any `forbidden_write_globs` path ← the #1 reward hack |
| `no_submit` | episode ended without `submit` |
| `sandbox_escape` | any write outside sandbox root |
| `syntax_broken` | repo no longer imports |

Log every gate rate on every rollout. Shaping terms (patch quality, tool efficiency, error recovery)
are **designed and documented in `docs/reward_design.md` but NOT implemented** — they're M7 ablations
in the training phase. Adding shaping before a clean baseline exists makes results uninterpretable.

---

# 6. Literature Review Brief (B5) ⭐

## 6.1 Rules
- **Verify every reference.** Retrieve the paper, confirm title / authors / venue / year / identifier.
  Anything unconfirmed gets `[UNVERIFIED — needs human check]`. Never invent a plausible-looking citation.
- Prefer primary sources over blog posts. Where you cite a blog or docs page, label it as such.
- Note the **date** of each source. This field moves fast; a 2023 claim about RL cost may be obsolete.
- For each paper: what problem, what method, what result, **and how it bears on our design**. A
  literature review that doesn't connect back to our choices is just a reading list.

## 6.2 Structure of `docs/literature_review.md`
1. **Problem framing** — reliability of small coding models on real software-engineering tasks
2. **RL for LLMs** — policy-gradient post-training; PPO → GRPO lineage; why GRPO's group-relative
   advantage removes the value model and cuts memory
3. **RLVR (verifiable rewards)** — outcome-based rewards from executable checks; why unit tests are
   an unusually clean reward source compared to preference models
4. **Coding agents & SWE benchmarks** — repo-level task benchmarks, agent harness design, patch
   evaluation methodology, contamination concerns
5. **Reward design & reward hacking** — outcome vs process rewards, shaping risks, specification
   gaming, why we gate rather than penalise
6. **Small models & efficiency** — what small open coding models can and can't do; parameter-efficient
   training; memory-efficient RL on constrained hardware
7. **Evaluation methodology** — pass@k estimation, sampling variance, why small benchmarks need CIs
8. **Gaps & positioning** — what is *not* well established, and where our feasibility study sits
9. **Bibliography** — verified entries only

## 6.3 Four design choices that must be defended against the literature
| Choice | Question the review must answer |
|---|---|
| GRPO over PPO/DPO | why group-relative advantage suits verifiable-reward settings and constrained memory |
| Test-pass-% as reward | outcome-based verifiable reward vs learned reward models — trade-offs |
| Hard-zero gates | what the literature says about specification gaming / reward hacking in code RL |
| pass@k with k=8 | sampling variance in small benchmarks; why pass@1 alone is not reportable |

## 6.4 Search strategy
Start from these anchors and follow citation graphs both directions:
- GRPO's originating work and its use in reasoning-model training
- RLVR / verifiable-reward post-training recipes
- Repo-level software-engineering benchmarks and their agent harnesses
- Qwen2.5-Coder technical reporting (capability baseline for our model class)
- Reward hacking / specification gaming literature
- pass@k estimation methodology from code-generation evaluation
- Memory-efficient RL and parameter-efficient fine-tuning for consumer hardware

Also survey **NeMo Gym's own environment hub** — if a suitable SWE-style environment already exists,
say so plainly. That would shift our contribution toward reward design and failure analysis, which is
arguably the stronger research angle and saves weeks. Record the finding in `docs/decisions/`.

---

# 7. `docs/TRAINING_READINESS.md` — the handoff doc ⭐

This is what makes the folder "pre-ready." Write it as if for someone with no memory of this build.

Required sections:
1. **What exists** — inventory: tasks by category/difficulty, environment components, reward version,
   metrics implemented, what is tested vs untested
2. **What was never run** — explicit list: no training, no full eval, no held-out access
3. **Exact next commands** — copy-pasteable, in order: serve model → smoke → baseline eval → smoke
   training run. Mark clearly which need hardware we don't have.
4. **Hardware decision table**

   | Path | Hardware | Status | Notes |
   |---|---|---|---|
   | Baseline eval | 4070 Ti 12 GB | ✅ ready | 1.5B via vLLM |
   | QLoRA GRPO (low-VRAM) | 4070 Ti 12 GB | ⚠️ untested | `grpo_lowvram.yaml`; assumptions documented |
   | NeMo RL GRPO | 8× 80 GB node | ❌ blocked | per official prerequisites |

5. **Blockers & options** — NVIDIA mentor compute request · university HPC · rented GPUs. State plainly
   that GRPO at the documented spec needs 8× 80 GB and our card is 12 GB.
6. **Known unknowns** — every assumption in the training configs that has never been executed
7. **VRAM levers** — the ordered list of knobs to turn on OOM: 4-bit load · LoRA rank ↓ ·
   `num_generations` 8→4 · `max_seq_length` ↓ · gradient accumulation ↑ · tool-output truncation ↑
8. **Risk register** — reward hacking, overfitting to 30 tasks, contamination, harness/NeMo-RL API drift

---

# 8. Quality Gates (CI — must be green at B6)

- [ ] All 40 tasks pass `task.schema.json`
- [ ] All 40: gold patch passes hidden tests
- [ ] All 40: untouched repo fails ≥1 hidden test
- [ ] `n_tests` in YAML matches actual collected count
- [ ] No task writes outside its `allowed_write_globs` under the gold patch
- [ ] Reward unit tests pass, including all four hard-zero gates
- [ ] `metrics.py` pass@k verified against hand-computed synthetic cases
- [ ] Sandbox isolation test: hidden tests unreadable and unwritable from inside
- [ ] `make smoke` completes one task end-to-end and returns a reward
- [ ] Fresh-clone test passes the §0.3 five-step check
- [ ] No secrets, weights, or >10 MB blobs in git history

---

# 9. Guardrails — never do these

1. **Never run RL training in this phase.** Ever. Under any justification.
2. Never let the agent read hidden tests.
3. Never open or peek at the held-out split after sealing.
4. Never write an unverified academic citation without marking it.
5. Never write code against remembered NeMo Gym / NeMo RL APIs.
6. Never commit secrets, checkpoints, or large binaries.
7. Never expand scope without approval.
8. Never edit this spec — propose changes in `docs/decisions/` and wait.
9. Never dress up an invented bug as a real-repo bug in `provenance`.
10. Never report a number that isn't reproducible from `results/`.

---

# 10. `PROGRESS.md` Convention

Append one block per session. Never edit past entries.

```markdown
## 2026-08-05 · B2 · Benchmark authoring
**Goal:** first 8 train tasks
**Done:** 8/30 authored (5 real_repo, 3 authored); all pass validate + gold-patch CI
**Commands:** `make validate` · `python scripts/verify_gold_patches.py --split train`
**Artifacts:** results/raw/b2_validate_20260805-1430/
**Numbers:** 8 tasks · 4 easy / 3 med / 1 hard · avg 140 LOC · avg 5.2 tests
**Blocked / needs human:** config_fix candidates feel too easy — make 2 multi-file?
**Next:** tasks 9–16, then review batch
```

Plus `docs/decisions/NNNN-title.md` for any non-obvious choice
(*Context → Options considered → Decision → Consequences*). Your supervisor and the NVIDIA mentor
**will** ask "why did you do it this way" — these files are the answer, written while you still remember.

---

# 11. Start Here

Execute in order. Stop after **B0** and report.

1. Scaffold the repo per §3; `uv` env on Python 3.12+
2. Record `nvidia-smi` → `docs/vram_notes.md`
3. Clone NeMo Gym; run **one built-in environment unmodified**
4. Write `docs/nemo_gym_notes.md` from the **real source code**, not memory
5. Commit; write the first `PROGRESS.md` entry
6. 🛑 **Stop. Report to human.**

Do not author tasks, design rewards, or touch training configs until B0 is signed off.
