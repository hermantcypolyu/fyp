# Phase 6F — correcting the defects the Phase 6E diagnostic smoke exposed

Phase 6E was a diagnostic smoke: four rollouts over one task, run with `--allow-uniform`
so that the artifacts would be written at all. Its own verdict was that infrastructure
acceptance **failed**, but it did produce the evidence this phase needed. Five defects were
found; all five are now fixed, and each fix has a test that fails without it.

Nothing in this phase started a model server, loaded a model, touched a GPU, ran a rollout,
a training step, or an evaluation. The five Phase 6E artifact files that describe the smoke
are byte-identical to their pre-phase hashes (see §7).

## 1. The uniform-reward guard destroyed the evidence it objected to

The guard against a constant reward ran **before** anything was written. When it fired,
roughly eighty seconds of GPU time left no rows on disk: the only surviving record was the
guard's own error message. The batch that most needs reading is precisely the one a guard
rejects, so the guard must never be the thing that deletes it.

The collector now runs four stages in order, and records each one separately in
`collection_summary.json`:

| key | meaning |
| --- | --- |
| `collection_completed` | the loop finished over every configured rollout |
| `evidence_persisted` | every public and private row reached disk and was `fsync`ed |
| `baseline_acceptance` | the run-level guards' verdict, with the problems listed |
| `training_suitability` | always `false` at this stage; stated, never inferred |

Persistence happens before any guard is consulted, so a rejected batch stays on disk and is
labelled as rejected. `_persist_evidence` reads back what it wrote and raises if the file
holds a different number of rows than were collected — a log that silently loses rows is
worse than one that refuses to report.

## 2. CLI overrides crashed on a frozen dataclass

`--diagnostic-only` and `--tokenizer` were applied by assignment:

```python
cfg.diagnostic_only = True  # FrozenInstanceError
cfg.tokenizer_path = args.tokenizer
```

The config is frozen on purpose: the record of what a run used must not be editable after
the run starts. Making the dataclass mutable would have fixed the crash and destroyed the
guarantee, so the fix builds a **new** instance instead, via `_with_overrides`, which writes
into a deep copy of the raw document and calls `dataclasses.replace`. The original object is
untouched and remains frozen.

## 3. `diagnostic_only` was a label, not a veto

The flag suppressed the context-overflow guard, which made a diagnostic run *easier* to
validate than a real one. It is now a structural veto in `_assess_validity`: a run can never
be both diagnostic-only and training-valid, whatever its numbers. A test asserts that a
clean run still passes, so the veto cannot be satisfied by a function that always fails.

## 4. Context telemetry was written in a shape the schema rejects, and to one file only

The schema's `context_budget` block is `additionalProperties: false` and declares
`capacity` / `peak_prompt_tokens`. The collector wrote `Probe.to_dict()`, whose keys are
different, and it wrote them only to the private sidecar — so nothing validated them, and
the mismatch hid itself: no field was ever written to the public row, so no field was ever
rejected.

`_public_context_telemetry` now maps the probe's values onto the schema's vocabulary and the
collector writes that record to both the public row and the sidecar, so the two describe one
measurement. The schema gained `status` and `context_remaining_tokens`, because "did this
rollout approach the wall" is the point of the record. A probe that failed before its first
measurement reports `peak_prompt_tokens: 0`; a rollout with no probe at all reports `null`,
since a zeroed record would claim a measurement that never happened.

The private sidecar also landed at mode `0664` in Phase 6E, because it came from a plain
`open()`. It is now written by `secure_write_private_log`, which creates it `0600`,
**narrows** a pre-existing file on resume, and verifies the resulting mode with `fstat`,
raising rather than returning if it is wrong.

## 5. The manifest reported the overlay's defaults as if they were the run's settings

Phase 6E's manifest recorded `rollouts_per_task: 8` and `tasks: null` — copied from the
pilot overlay — while the run executed four rollouts over one task. Nothing in the file let
a reader tell "the overlay says 8" from "the run used 8", which is how a default silently
impersonates a measured value.

`build_manifest` now writes an `effective` block alongside the file records:

- **resolved values** — `rollouts_per_task`, `tasks`, `concurrency` as the run used them;
- **three hashes** — base protocol, overlay, and the merged document (canonical JSON,
  sorted keys, so the fingerprint depends on the settings rather than on YAML ordering);
- **per-field provenance** — a dotted path to the layer that supplied it.

The existing `pilot` block keeps the overlay file's own values. The two blocks are
deliberately different and now distinguishable, which is exactly what Phase 6E could not do.

## 6. The serving-venv import shadowing, and why the fix is a package

`scripts/collect_baseline_rollouts.py` drives a vLLM server, so it must run under the
serving virtualenv (`~/.venvs/qwen-server`), which pins `transformers` 4.45.2. That venv
contains a **regular** package at `site-packages/scripts/` (shipped by llama-cpp-python),
while this repository's `scripts/` was an implicit **namespace** package. A regular package
beats a namespace package regardless of `sys.path` order, so `import scripts.x` resolved to
llama-cpp's empty package and the collector failed at import.

**Decision B was taken:** add `scripts/__init__.py`, making both candidates regular
packages, so `sys.path` order decides — and the repository, placed first, wins.

**Decision A is also present, as the documented way to run it.**
`scripts/run_in_serving_env.py` validates the environment before exec'ing a script (correct
interpreter, required package versions, and that `scripts` resolves inside the repository),
then sets a repository-first `PYTHONPATH`. It needs no shim, and it fails with a clear
message rather than a confusing `ImportError` when the environment is wrong.

The earlier `/tmp/phase6e_shim` workaround is no longer needed by anything and is not
referenced by any code path.

## 7. Verification

| check | result |
| --- | --- |
| `make check` | 639 passed, gold 36/36, ruff clean, format clean |
| new tests | 25, all CPU-only, in `analysis/tests/test_phase6f_defects.py` |
| held-out seal | `10 tasks, tree 318fedc8868b unchanged` |
| `git diff --check` | no whitespace errors |
| GPU | both cards idle; no server started, no model loaded |

Phase 6E artifacts in `results/raw/phase6e_bugfix_offbyone_pagination_20261105T193011Z/`
are unchanged:

| file | bytes | lines | sha256 (16) |
| --- | --- | --- | --- |
| `rollouts.jsonl` | 76879 | 4 | `705d5cc64bad7b63` |
| `rollouts_private.jsonl` | 36192 | 4 | `923ab940e04f9605` |
| `rollouts_failures.jsonl` | 0 | 0 | `e3b0c44298fc1c14` |
| `collection_summary.json` | 1137 | 37 | `a9d5a6d19d64c3d1` |
| `phase6e_merged_config.yaml` | 5417 | 181 | `37dfefe65950b662` |

`manifest.json` differs (5727 B, `ab2fb75badc4f7b6`) because it was deliberately closed
afterwards with `run_manifest.py --finish`; the change is confined to timing, outcome, and
checkpoints, which is what `--finish` is permitted to touch.

## 8. What these fixes do not claim

- **Phase 6E remains a failed infrastructure acceptance.** These fixes make the failures
  visible and non-destructive; they do not make the four rollouts training evidence. The
  batch was diagnostic-only and its `training_suitability` is `false` by construction.
- **A passing `make check` is not a passing baseline.** The unit tests exercise the
  collector's guarantees, not a model. A real 1.5B baseline still has to be collected and
  pass acceptance on its own numbers.
- **The model, protocol, reward, and data are untouched.** No sampling parameter, task
  split, verifier, held-out task, or checkpoint was changed in this phase.
