# Handoff / Reproducibility Checklist

This file documents the exact, minimal steps needed to reproduce a development environment and continue work on another machine.

Prerequisites
- Git installed and configured with SSH keys for `git@github.com:hermantcypolyu/fyp.git` (or HTTPS remote of your choice).
- Python 3.12 installed.
- `uv` CLI (optional but recommended) for deterministic editable installs and sync.

Quick transfer steps (new machine)
1. Clone the repo:

```bash
git clone git@github.com:hermantcypolyu/fyp.git
cd fyp/rl-coding-agents
```

2. Create and activate environment (recommended using `uv`):

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
```

If you don't use `uv`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]   # or use pyproject tooling
```

3. Create machine-local secrets file:

```bash
cp env.example.yaml env.yaml
# edit env.yaml to add API keys and provider endpoints
```

4. Run quick checks:

```bash
make validate
make test
make smoke
```

Developer workflow notes
- Use `git checkout -b feat/whatever` for new work.
- Run `make test` before committing.
- Keep `env.yaml`, `.venv/`, `_vendor/` (if present), `results/` and other large files out of the repository (see `.gitignore`).

Large files & artifacts
- Host large model weights or datasets in external storage (S3/GCS) and add downloader scripts under `scripts/` rather than committing binary blobs.
- If you must track large files, use Git LFS and document tracked extensions in `.gitattributes`.

If anything in these instructions fails on a new machine, open an issue or contact the original developer with the console output and the contents of `env.yaml` (redacted of secrets).

-- End of Handoff
