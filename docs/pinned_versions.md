# Pinned third-party versions

Both stacks are pre-1.0 and move fast. Every line of code in this repo that touches them was
written against the commits below, with the source open (spec §1.3) — not from remembered APIs.

The clones live in `../_vendor/`, **outside** the repository and gitignored, so nothing here
depends on them being present: `make validate`, `make test` and `make smoke` all work without them.

| Stack | Commit | Committed | `requires-python` | Used for |
| --- | --- | --- | --- | --- |
| [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) | `ec5a8136046aa0dd0c1525313654502f2fb7c42e` | 2026-07-31 | `>=3.12` | resources server, agent harness, `/verify` contract |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) | `55296257c96d49cd95c7d77613cb0f36bd4a4dc7` | 2026-08-02 | `>=3.13.13,<3.14` | **reference only** — GRPO trainer for the training phase |

NeMo RL is cloned and read, **not installed and not run**. Nothing in this repo imports it.

## ⚠️ The two stacks do not share a Python version

This is the single most important fact on this page, and it is easy to miss:

- NeMo Gym runs on **3.12** here (and must: `ray` publishes no `win_amd64` wheel past cp312).
- NeMo RL requires **3.13.13 ≤ python < 3.14** — it will refuse to install on 3.12.

So a training box needs **two interpreters**, not one. NeMo RL manages its own `uv`-locked
environment; do not try to install it into this project's `.venv`. See
[SETUP_NEW_MACHINE.md](SETUP_NEW_MACHINE.md) §3.

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

## Re-pinning

1. `git -C ../_vendor/<dir> fetch && git -C ../_vendor/<dir> checkout <new-sha>`
2. Update `PINNED` in [analysis/versions.py](../analysis/versions.py) **and this table**
   (a test asserts they agree).
3. Re-read the built-ins the code was adapted from — the `# Pattern adapted from:` comments in
   `envs/coding_env/` name the exact files.
4. Re-run `make test` and `make smoke`.
