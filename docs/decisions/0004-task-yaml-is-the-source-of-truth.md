# 0004 — `task.yaml` is the source of truth; the agent-facing JSONL is a build artefact

- **Status:** Accepted
- **Date:** 2026-07-31
- **Milestone:** B0 (implemented in B1)

## Context

Two formats want to describe the same 40 tasks:

- **`benchmark/tasks/<split>/<id>/task.yaml`** — the human-authored contract of §5.1: instruction,
  verification block, constraints, provenance, metadata.
- **A NeMo Gym dataset JSONL** — one row per task, carrying `responses_create_params` with the
  prompt and the OpenAI function-tool schema for all 6 tools (see `docs/nemo_gym_notes.md` §4).

The tool schema is *identical for every task* (§5.5 fixes the surface at 6 tools). Duplicating it
into 40 JSONL rows by hand means 40 places to update when a tool signature changes, and 40 chances
for a row to drift out of sync with the server that actually implements it.

## Options considered

1. **Author the JSONL by hand, treat YAML as documentation.** Rejected: the JSONL is
   single-line-per-task, unreviewable in a diff, and impossible to comment. Provenance and licence
   fields (§5.3) would rot.
2. **Author YAML, and have the resources server read `task.yaml` at request time.** Rejected: it
   couples the runtime to the on-disk layout and makes the dataset non-portable — you could not
   hand someone a JSONL and have it work.
3. **Author YAML; generate the JSONL with `benchmark/tools/build_catalogue.py`.** ✅ Chosen.

## Decision

```
benchmark/tasks/<split>/<task_id>/task.yaml     # authored, reviewed, git-diffable
        │
        ├── benchmark/schema/validate_tasks.py  # jsonschema + cross-checks  (make validate)
        │
        └── benchmark/tools/build_catalogue.py  # generator                  (make catalogue)
                 ├── benchmark/data/<split>.jsonl   # agent-facing, regenerable
                 └── benchmark/catalogue.md         # human-facing table
```

Rules that follow:

- **Generated files carry a header** naming the generator and stating that edits will be
  overwritten. `benchmark/data/*.jsonl` is regenerable at any time from YAML.
- **The tool schema is emitted from one Python definition**, the same object the resources server
  registers its routes from. A tool cannot be advertised to the model but missing from the server.
- **`make validate` is the gate**, and CI-style checks run on YAML, not JSONL: schema conformance,
  `id` matches directory name, referenced `test_files` exist, `split` matches parent directory,
  glob constraints are non-empty, provenance present.
- **Hidden tests never enter the JSONL.** The prompt is built from `instruction` plus the file
  listing; `verification.test_files` stays server-side only. This is the §5.4 hidden-test rule
  enforced at the format level rather than by reviewer discipline.

## Consequences

- ✅ One edit changes a tool signature everywhere; no drift between dataset and server.
- ✅ Tasks review like code: YAML diffs are readable, provenance and licence are visible.
- ✅ Anyone can regenerate the dataset; nothing hand-maintained can silently rot.
- ✅ Hidden tests cannot leak into the prompt by accident — the generator has no path to them.
- ⚠️ You must remember to run `make catalogue` after editing YAML. Mitigated by making `make check`
  regenerate and then fail if the working tree is dirty.
- ⚠️ Generated JSONL is committed (so a fresh clone can `make smoke` without a build step), which
  means generator changes produce large diffs. Accepted: the files are small and the header says
  not to hand-edit them.
