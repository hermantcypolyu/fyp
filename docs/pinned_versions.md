# Pinned third-party versions

Both stacks are pre-1.0 and move fast. Every line of code in this repo that touches them was
written against the commits below, with the source open (spec §1.3) — not from remembered APIs.

The clones live in `../_vendor/`, **outside** the repository and gitignored, so nothing here
depends on them being present: `make validate`, `make test` and `make smoke` all work without them.

| Stack | Commit | Committed | `requires-python` | Used for |
| --- | --- | --- | --- | --- |
| [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) | `ec5a8136046aa0dd0c1525313654502f2fb7c42e` | 2026-07-31 | `>=3.12` | resources server, agent harness, `/verify` contract |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) | `55296257c96d49cd95c7d77613cb0f36bd4a4dc7` | 2026-08-02 | `>=3.13.13,<3.14` | **reference only** — GRPO trainer for the training phase |

NeMo RL is cloned and read, **not installed and not run** by this project. Nothing in this repo
imports it. It maintains its own `uv`-locked environment inside `../_vendor/nemo-rl/`.

## ⚠️ The two stacks do not share a Python version

This is one of the two things on this page that is easy to miss:

- NeMo Gym declares `>=3.12`. The original constraint to exactly 3.12 was a **Windows** one:
  `ray` publishes no `win_amd64` wheel past cp312. On Linux that constraint does not bind —
  see "Observed on Linux" below.
- NeMo RL requires **3.13.13 ≤ python < 3.14** — it will refuse to install on 3.12. Its
  `.python-version` reads `3.13.13`.

So a training box needs **two interpreters**, not one. NeMo RL manages its own `uv`-locked
environment; do not try to install it into this project's `.venv`. See
[SETUP_NEW_MACHINE.md](SETUP_NEW_MACHINE.md) §3.

### Observed on Linux (2026-08-03)

On the A5000 box the project `.venv` resolved to **3.13**, and the `gym` extra installed and
ran correctly there (`ray==2.56.1`, `uvloop==0.22.1`, `nemo-gym` importable, `make smoke-gym`
green). The 3.12 pin is therefore a *Windows-only* requirement, not a NeMo Gym one. Recorded
here so the next reader does not force 3.12 on Linux for a reason that does not apply.

---

## ⚠️ Pinning the source clone does not pin the imported artifact

This is the second easy-to-miss fact, and it defeated the drift check on the first Linux run.

`analysis/versions.py` reads the `HEAD` of each **vendored clone**. It says nothing about which
`nemo_gym` Python package the interpreter actually *imports*. Those are two different things,
and on 2026-08-03 they disagreed while the drift check reported `matches: true`:

| | Resolves to | Contains `ReverifyMode`? |
| --- | --- | --- |
| `../_vendor/nemo-gym` @ `ec5a8136` | `0.5.0rc0` | ✅ yes (`base_resources_server.py:71`) |
| PyPI `nemo-gym`, via `uv sync --extra gym` | `0.4.0` | ❌ no |

`envs/coding_env/app.py:36` imports `ReverifyMode` and line 52 sets
`REVERIFY_MODE = ReverifyMode.UNSUPPORTED`. Against the PyPI wheel that is an `ImportError`, so
`--transport gym` cannot start at all:

```
cannot import name 'ReverifyMode' from 'nemo_gym.base_resources_server'
    (.venv/lib/python3.13/site-packages/nemo_gym/base_resources_server.py)
```

`ReverifyMode` entered upstream in `da0caf6f` (*Rerun judge failures with reverify*, #2156),
which is included in the pinned commit but not in any published release. `0.5.0rc0` is not on
PyPI, so `gym = ["nemo-gym>=0.4.0"]` in `pyproject.toml` can only ever resolve to `0.4.0`.

### The fix: install the vendored clone, not the wheel

```bash
make gym-vendor
```

which is:

```make
gym-vendor:
	uv sync --extra gym
	uv pip install --no-deps -e ../_vendor/nemo-gym
```

`uv sync --extra gym` still does the dependency resolution (`uvloop`, `ray`, `fastapi`, …);
`--no-deps -e` then replaces the *code* with the pinned source. Confirm which one is loaded:

```bash
uv run python -c "import inspect, nemo_gym; print(inspect.getfile(nemo_gym))"
```

It must print a path under `../_vendor/nemo-gym/`, not under `site-packages/`.

### ⚠️ `uv sync` silently undoes this

Observed, not theorised — every `uv sync --extra gym` reinstalls the PyPI wheel over the
editable install:

```
- nemo-gym==0.5.0rc0 (from file:///.../_vendor/nemo-gym)
+ nemo-gym==0.4.0
```

The next `--transport gym` run then fails with the `ImportError` above. **Any `uv sync` must be
followed by `make gym-vendor`.** That is why the target does both steps in order.

This is a gap in the drift mechanism, not a bug in it: the trace would have recorded
`matches: true` for a run executing entirely different code. Recording the imported
distribution's version and file path alongside the clone `HEAD` would close it.

## Drift is detected, not assumed

`analysis/versions.py` holds these commits as `PINNED` and reads the *actual* `HEAD` of each
vendored clone at runtime. Every rollout trace records both plus a `matches` boolean
(`analysis/trace_schema.json` → `versions`), so a result produced against a drifted checkout is
visible in the data rather than inferred later:

```json
"nemo_gym": {
  "pinned": "ec5a8136046aa0dd0c1525313654502f2fb7c42e",
  "actual": "ec5a8136046aa0dd0c1525313654502f2fb7c42e",
  "matches": true
}
```

Recording only the pin would let a clone drift silently; recording only the actual commit would
make an old result impossible to interpret. Both, or neither is worth much.

### It fired on the first Linux run

A fresh `git clone` of each stack on 2026-08-03 landed on `main`, not on the pin, and the check
caught both before anything was run:

| Stack | Pinned | Fresh clone `HEAD` | |
| --- | --- | --- | --- |
| NeMo Gym | `ec5a8136` | `03efc8c5` | ❌ drifted |
| NeMo RL | `55296257` | `8fb82ca8` | ❌ drifted |

Both were resolved with `git checkout <pinned-sha>` (detached `HEAD` is expected and correct
for a vendored pin), followed by `git submodule update --init --recursive` for NeMo RL, whose
`3rdparty/` pins move with the branch. `resolve_all()` then returned `matches: true` for both.

Two lessons worth stating plainly: `git clone` alone does **not** reproduce a pinned checkout,
and the clone must live where `versions.py` expects it (`../_vendor/<dir>`) or `_head()`
returns `None` and `matches` silently becomes `False`.

## Re-pinning

1. `git -C ../_vendor/<dir> fetch && git -C ../_vendor/<dir> checkout <new-sha>`
2. For NeMo RL only: `git -C ../_vendor/nemo-rl submodule update --init --recursive` —
   different commits pin different `3rdparty/` versions.
3. Update `PINNED` in [analysis/versions.py](../analysis/versions.py) **and this table**
   (a test asserts they agree).
4. Re-read the built-ins the code was adapted from — the `# Pattern adapted from:` comments in
   `envs/coding_env/` name the exact files.
5. Re-run `make gym-vendor` — a re-pin is a `uv sync`, and a `uv sync` reverts the editable
   install to the PyPI wheel.
6. Re-run `make test`, `make smoke` and `make smoke-gym`.

## Verification log

| Date | Box | Result |
| --- | --- | --- |
| 2026-08-03 | A5000 / WSL2 Ubuntu | Both clones re-pinned; `resolve_all()` → `matches: true` for both. Vendored `nemo-gym` install required; `make smoke` and `make smoke-gym` both score `1.00`. `make check` green: 36/36 tasks, 232 tests passed, held-out tree `318fedc8868b` unchanged. |