# Setting up on a new machine

Cold start to a verified benchmark in five steps, then the extra work a Linux/GPU box needs.

Everything in **§1–§5 runs on any OS with no GPU and no NeMo stack**. That is deliberate
(ADR 0002): the benchmark and the environment must be checkable without a training rig, or the
only person who can validate this work is the person holding the hardware.

---

## 0. What you need

| | Requirement |
| --- | --- |
| Python | **3.12** for this repo and NeMo Gym. **Not 3.14** — see §1. |
| Tooling | [`uv`](https://docs.astral.sh/uv/) ≥ 0.12, `git`, `make` (optional — `make.ps1` mirrors it on Windows) |
| Disk | ~1 GB for the repo + vendored clones; ~3.5 GB more if you download model weights |
| Network | `git clone` and PyPI. Corporate TLS interception → set `UV_SYSTEM_CERTS=1` |

---

## 1. Python 3.12 — not 3.14, not 3.13

```bash
uv python install 3.12
uv venv --python 3.12
```

Three independent reasons, none of them preference:

1. **`ray` has no `win_amd64` wheel past cp312**, and NeMo Gym's orchestration imports it.
2. **NeMo Gym declares `requires-python = ">=3.12"`** and its own venv is 3.12.10.
3. **3.14 breaks the toolchain today.** This project was *developed* on 3.14 because that was
   what the machine had, and it works for `validate` / `test` / `smoke` — but the NeMo stack
   does not install there. Do not repeat that constraint if you have a choice.

> **NeMo RL is different and needs its own interpreter: `>=3.13.13,<3.14`.** It will not install
> into a 3.12 environment. See §3 and [pinned_versions.md](pinned_versions.md).

---

## 2. This repo, from the committed lock

```bash
git clone <this-repo> rl-coding-agents
cd rl-coding-agents
export UV_SYSTEM_CERTS=1          # only if your network intercepts TLS
uv sync --extra dev               # resolves from the committed uv.lock - do not re-resolve
```

`uv.lock` is committed on purpose. `uv sync` reproduces the exact dependency set; `uv lock
--upgrade` does not, and a silently upgraded pytest changes what "12 of 12 passed" means.

Core dependencies are three pure-Python packages (`pyyaml`, `jsonschema`, `numpy`) so this step
takes under a minute and needs no compiler.

---

## 3. Vendored clones — pinned, outside the repo

Both live in `_vendor/`, a **sibling of** the repo, and are gitignored. Nothing in §5 needs them.

```bash
mkdir -p ../_vendor && cd ../_vendor

git clone https://github.com/NVIDIA-NeMo/Gym.git nemo-gym
git -C nemo-gym checkout ec5a8136046aa0dd0c1525313654502f2fb7c42e

git clone https://github.com/NVIDIA-NeMo/RL.git nemo-rl
git -C nemo-rl checkout 55296257c96d49cd95c7d77613cb0f36bd4a4dc7
```

**NeMo RL is a reference, not a dependency.** Do not install it and do not run it in this phase.
Nothing in this repository imports it; it is cloned so that the GRPO trainer we will eventually
target is pinned to a commit we have actually read.

To install NeMo Gym (Linux/macOS, optional — only `make smoke-gym` needs it):

```bash
cd ../rl-coding-agents
uv sync --extra gym
```

On Windows this needs the two workarounds recorded in
[nemo_gym_notes.md §7](nemo_gym_notes.md): a `resource` shim on `PYTHONPATH` and a `uvloop`
override. Neither modifies vendored source.

---

## 4. Model weights (only for eval/training — §5 does not need them)

```bash
uv pip install "huggingface_hub[cli]"
hf download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir ../_models/qwen2.5-coder-1.5b-instruct
```

~3.1 GB in bf16. `_models/` is outside the repo; weights are never committed (spec §1.5, and
`.gitignore` blocks `*.safetensors`, `*.bin`, `checkpoints/`, `models/`).

Serving it (vLLM, ~4 GB VRAM at bf16 with a short context) is a §6 concern.

---

## 5. Verification sequence — run these in order

```bash
uv run python benchmark/schema/validate_tasks.py --all       # 1. corpus is well-formed
uv run python scripts/verify_gold_patches.py --split train   # 2. every task is solvable AND failable
uv run pytest                                                # 3. unit + isolation suites
make smoke                                                   # 4. one task end-to-end -> a real reward
```

What each one actually proves, and what a failure means:

| Step | Expected | If it fails |
| --- | --- | --- |
| `validate_tasks --all` | `26 task(s), 0 errors` (5 mix/provenance warnings are known and expected) | The corpus is malformed. `--fix-counts` rewrites the two measured fields (`repo.loc`, `verification.n_tests`); everything else is a real authoring error. |
| `verify_gold_patches` | `26/26 solvable and failable`, mean floor ≈ 0.46 | A gold patch no longer applies, or a hidden suite drifted. Stop — every reward downstream is meaningless until this is green. |
| `pytest` | `147 passed, 1 skipped` | The skip is `test_modules_import_when_the_gym_stack_is_present`, expected without `--extra gym`. Anything else is a regression. |
| `make smoke` | `PASS … reward=1.00 tests=12/12 gates=-` | The tool surface, sandbox, write globs or verifier is broken. No prompt engineering will fix a broken environment. |

With NeMo Gym installed, also run the HTTP path — it exercises `app.py`, the session middleware
and `POST /verify`, which the four steps above never touch:

```bash
make smoke-gym     # scripts/smoke_one_task.py --limit 1 --transport gym
```

Both transports must report the **same** reward. §5.6 requires evaluation and the RL reward to
come from one verifier, and this is how that is checked rather than assumed.

To capture a trace while you are there:

```bash
uv run python scripts/smoke_one_task.py --limit 1 --trace results/raw/setup_check/rollouts.jsonl
```

Rows are validated against [`analysis/trace_schema.json`](../analysis/trace_schema.json) as they
are written, so an invalid trace fails immediately instead of during analysis.

## 5a. Linux / GPU prerequisites and vendoring gotchas (read this closely)

On a box with a GPU the following items are prerequisites or common traps; they are
operational rules, not opinions, because they change what code actually runs.

- WSL2 with working GPU passthrough is a hard prerequisite. `nvidia-smi` must
   succeed inside the Ubuntu session where you run `uv` and `make` — seeing it in
   PowerShell is not enough. If `nvidia-smi` fails in the WSL shell, CUDA and the
   rest of the stack will appear absent at runtime.

- Do *not* use the system Python. This box had Python 3.14 installed system-wide and
   PyTorch publishes no CUDA wheels for cp314: `pip install torch` therefore silently
   yields a CPU-only build and `torch.cuda.is_available()` returns `False` with no
   error. Let `uv` manage interpreters and venvs; NeMo RL pins `3.13.13` in its
   `.python-version` and must run in the interpreter it expects.

- Keep the repository under your home directory, not under `/mnt/c` or any Windows
   mount. The verifier copies a repo and runs `pytest` for every graded rollout; the
   Windows filesystem through WSL makes those copies and tests unacceptably slow.

- Files that arrive via Windows carry CRLF. Strip CRLF before committing. The
   pristine-source verifier compares against untouched source and CRLF makes every
   file look modified; run `dos2unix` or `git restore --source=HEAD --worktree --` on
   affected files before committing.

- Vendored clones must live at `../_vendor/<dir>` exactly. `analysis/versions.py`
   resolves those paths; if it cannot find the clone `_head()` returns `None` and the
   `matches` boolean can become `False` silently. Do not move or rename the clones.

- A fresh `git clone` lands on `main`, not on the pinned SHA. After checking out a
   pinned commit you must, for NeMo RL only, run `git submodule update --init --recursive`
   because the `3rdparty/` pins move with the branch.

- Install `nemo-gym` from the vendored clone, not from PyPI. The published `0.4.0`
   wheel lacks `ReverifyMode`; the vendored clone is `0.5.0rc0` (unreleased) and
   contains the code this repo imports. Use `make gym-vendor` which performs

```bash
uv sync --extra gym
uv pip install --no-deps -e ../_vendor/nemo-gym
```

   Note: `uv sync --extra gym` reinstalls the PyPI wheel over an editable install.
   Always run `make gym-vendor` after any `uv sync` so the editable vendored clone
   is restored.

- Two artefact classes are intentionally local-only and must be restored by hand from
   the USB backup: everything matching `results/**/*.jsonl` and the file
   `.git/info/exclude`. On a fresh machine place them at the exact paths under the
   restored checkout (`results/` in the repo root, and `.git/info/exclude` inside the
   repo's `.git` directory). These are excluded from the bundle on purpose.

- Both pinned SHAs live in `docs/pinned_versions.md`; do not duplicate them here —
   point readers at that file for the authoritative source of truth.


---

## 6. ⚠️ REQUIRED on Linux: re-check the resource limits

**Do this before any training run. It has never been exercised.**

`ResourceLimits` (in [`envs/coding_env/limits.py`](../envs/coding_env/limits.py)) caps the graded
pytest subprocess. Four of its six caps are POSIX `setrlimit` calls, and this project was built on
Windows, where `setrlimit` does not exist. On the development machine every rollout trace honestly
records:

```json
"limits_enforced": {
  "wall_clock_s": true,  "network": true,
  "cpu_s": false, "memory_mb": false, "max_file_mb": false, "max_processes": false
}
```

So the wall-clock timeout and the network guard are **tested**; the CPU, memory, file-size and
process caps are **written but never run**. `preexec_fn()` returns `None` off POSIX, so the code
path has never executed even once.

```bash
uv run pytest envs/tests/test_verifier.py -q          # limits tests now assert the POSIX branch
uv run python -c "from envs.coding_env.limits import ResourceLimits as R; print(R().enforced())"
```

Expected on Linux: **all six `true`**. If any is `false`, `os.name != 'posix'` is not the reason and
something is wrong.

Then confirm the caps actually bite, because "the call did not raise" is not the same as "the limit
applied". Suggested checks, each expected to fail the run rather than hang the host:

| Cap | Probe |
| --- | --- |
| `memory_mb` | a task whose repo allocates `bytearray(4 * 1024**3)` → `MemoryError` |
| `cpu_s` | a busy loop → `SIGXCPU` before the wall clock |
| `max_file_mb` | writing a 1 GB file → `OSError` |
| `max_processes` | a fork bomb → `BlockingIOError` |

Record the outcome in `docs/vram_notes.md` alongside the GPU findings. Until that is done, treat
`memory_mb` and friends as **claimed, not enforced** — that is why `limits_enforced` is on every
trace row rather than in a config file nobody reads.

Two further Linux-only gaps from earlier milestones:

- `gym env start` boots a Ray cluster, which **cannot start on native Windows**
  ([nemo_gym_notes.md §7](nemo_gym_notes.md)). The multi-process orchestration path is unverified.
- The network guard is a `sitecustomize.py` monkey-patch, not a firewall. If the training box has
  a container runtime, prefer a network-namespace-isolated sandbox and demote the guard to
  defence-in-depth.

---

## 7. Hardware

Measured on the development machine unless marked *projected*. **There is no NVIDIA GPU here**
(ADR 0001), so every GPU figure below is projected from published model sizes, not observed.

| Workload | Needs | Status |
| --- | --- | --- |
| `validate_tasks`, `verify_gold_patches`, `pytest`, `make smoke` | CPU only, < 1 GB RAM, ~40 s total | **measured** |
| `make smoke-gym` (HTTP path, in-process TestClient) | CPU only | **measured** (Windows, via the shim) |
| Serving Qwen2.5-Coder-1.5B-Instruct, bf16, short context (vLLM) | 1 GPU, **~4–6 GB VRAM** | *projected* — 1.5B × 2 bytes ≈ 3.1 GB weights + KV cache |
| Baseline eval, 26 tasks × k samples | 1 GPU, **12 GB is enough** | *projected* |
| GRPO with NeMo RL, 1.5B policy, full precision optimiser | **8 × 80 GB** (the spec's target) | *projected*, never run |
| GRPO fallback on 12 GB (QLoRA, tiny group size) | 1 GPU, 12 GB | *projected*; documented in `configs/training/grpo_lowvram.yaml` when B6 writes it |

Cap any smoke test at **3 tasks × 1 sample** (spec §1.4). On CUDA OOM, do **not** silently shrink
settings and retry — record it in `docs/vram_notes.md`. That file is the evidence base for the
training-phase config.

---

## 8. Things that will bite you

1. **`.venv/` and `_vendor/` are gitignored.** If `git status` ever shows them, something rewrote
   `.gitignore`.
2. **Do not open `benchmark/tasks/heldout/`.** It is authored to the train distribution and sealed;
   reading it during development is how a held-out split stops being held out.
3. **`benchmark/tasks/**` is excluded from ruff and from pytest collection.** Those repos are
   deliberately buggy — they are data, not source.
4. **Hidden suites must never be run by our own pytest.** `norecursedirs` blocks it; they are
   executed only through `benchmark.tools.task_loader`, with a generated ini.
5. **Windows line endings.** `.gitattributes` pins LF; the `CRLF will be replaced by LF` warnings
   on commit are expected.
