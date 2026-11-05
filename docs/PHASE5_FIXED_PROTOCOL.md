# Phase 5 — The fixed experiment protocol

**Status: the protocol is written down and pinned. Nothing here has been executed.**

Phase 5 exists for one reason: to make the sentence *"RL training changed the score"* mean
something. For that sentence to be true, everything except the treatment has to be identical
between the two arms, and every run has to say which values it used. This document records the
fixed settings, the manifest that carries them, and the machine rules that keep the GPU usable
for the next run.

| Sub-phase | What it pins | Where it lives |
| --- | --- | --- |
| 5.1 | every setting the two arms share | `configs/training/protocol_1p5b.yaml` |
| 5.2 | what each run records about itself | `scripts/run_manifest.py` |
| 5.3 | what a session may do to the GPU | [PHASE5_GPU_RESOURCE_RULES.md](PHASE5_GPU_RESOURCE_RULES.md) |

**Provenance vocabulary.** Every value below is tagged:

- **GROUNDED** — copied from a config that already exists in this repository.
- **CHOSEN** — a decision made here, with a reason, not derivable from an existing file.
- **ASSUMPTION** — accepted as true without verification on hardware.
- `null` — **undecided**, and reported as a gap rather than filled with a plausible number.

---

## 1. The two arms

| Arm | Config | What it is |
| --- | --- | --- |
| Control | `configs/eval/baseline.yaml` | base 1.5B model, no adapter, sampled and graded |
| Treatment | `configs/training/protocol_1p5b.yaml` + `configs/training/grpo_nemo_rl_a5000.yaml` | the same model after GRPO + LoRA |

`configs/training/grpo_unsloth_a5000.yaml` carries `status: superseded` and
`superseded_by: configs/training/grpo_nemo_rl_a5000.yaml`. It is kept because it is the
upstream source of the LoRA and optimiser shapes — the NeMo RL config re-expresses a plan that
was first written for Unsloth — but it is **not** an arm, and nothing is asserted against it.

The protocol file is the arm-independent half of the treatment. The trainer config is the
trainer-specific half. Splitting them this way means the shared settings can be hashed and
compared across both arms, which is what makes a parity test possible.

---

## 2. Phase 5.1 — the fixed settings

All twenty checklist items are present in `configs/training/protocol_1p5b.yaml` and asserted
present by `analysis/tests/test_protocol_parity.py::test_protocol_declares_every_phase_5_1_setting`
(the list is `PHASE_5_1_FIELDS`, which is the authority — count it rather than trusting this
sentence).

### 2.1 Model

| Setting | Value | Provenance | Why this value |
| --- | --- | --- | --- |
| Base model | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | GROUNDED | same string in both arms; asserted by `test_protocol_names_the_model_the_arms_actually_load` |
| Revision | `2e1fd397ee46e1388853d2af2c993145b0f1098a` | GROUNDED | the cached snapshot's commit sha; resolved in Phase 5.4 (§2.5) |
| Quantisation | `load_in_4bit: true` | GROUNDED | `baseline.yaml`; 4-bit vs 16-bit is a larger effect than RL |
| Full finetuning | `false` | GROUNDED | 24 GB does not fit a full 1.5B update with a vLLM server resident |
| Max sequence length | `8192` | GROUNDED | `baseline.yaml`; the two arms must see the same context window |

### 2.2 LoRA

| Setting | Value | Provenance | Why this value |
| --- | --- | --- | --- |
| `r` | `32` | GROUNDED | from the Unsloth plan, preserved by the NeMo RL config |
| `lora_alpha` | `64` | GROUNDED | `2 × r`, the conventional scaling |
| `target_modules` | `q, k, v, o, gate, up, down` | CHOSEN | all seven; dropping one silently trains less of the model than reported |
| `random_state` | `42` | CHOSEN | initialisation seed; without it "same config" can mean different adapters |

`test_lora_targets_cover_both_projections_of_every_layer` pins the full list, because a
truncated list is the quietest possible way to under-train.

### 2.3 Training

| Setting | Value | Provenance | Why this value |
| --- | --- | --- | --- |
| `learning_rate` | `1.0e-5` | GROUNDED | both trainer configs agree |
| `weight_decay` | `0.001` | GROUNDED | Unsloth plan, carried forward |
| `warmup_ratio` | `0.0` | GROUNDED | no warmup in the existing plan |
| `lr_scheduler_type` | `linear` | GROUNDED | as above |
| `optim` | `adamw_8bit` | CHOSEN | 24 GB budget; 32-bit Adam states are a large fraction of a 1.5B train |
| `max_grad_norm` | `1.0` | CHOSEN | standard clipping; recording it prevents "clipped or not" being unknowable later |
| `per_device_train_batch_size` | `1` | GROUNDED | 24 GB |
| `gradient_accumulation_steps` | `16` | GROUNDED | effective batch 16 |
| `training_iterations` | `5` | CHOSEN | a smoke-scale number, not a target-quality number |
| `checkpoint_every_n_iterations` | `1` | CHOSEN | with 5 iterations, every-iteration checkpoints cost little and permit §4 reload checks |

`training_iterations: 5` is deliberately small. Phase 5 is about *attributability*, not about
producing the best adapter; a five-iteration run is enough to show the pipeline turns and
produces a reloadable checkpoint, and small enough that a mistaken launch costs minutes.

### 2.4 Rollouts, sampling and episode budget

| Setting | Value | Provenance | Why this value |
| --- | --- | --- | --- |
| `rollouts.group_size` | `4` | GROUNDED | `num_generations_per_prompt: 4` in `grpo_nemo_rl_a5000.yaml` |
| `rollouts.n_tasks` | `4` | GROUNDED | `num_prompts_per_step: 4` |
| `rollouts.rollouts_per_task` | `4` | GROUNDED | equals group size; asserted |
| `sampling.temperature` | `1.0` | GROUNDED | `baseline.yaml`; asserted equal across arms |
| `sampling.top_p` | `1.0` | GROUNDED | both arms agree; resolved in Phase 5.4 (§2.5) |
| `sampling.seed` | `42` | GROUNDED | both arms |
| `episode.max_turns` | `from_task_yaml` | GROUNDED | the task's own budget is authoritative |
| `episode.max_tool_calls` | `from_task_yaml` | GROUNDED | as above |

### 2.5 The one remaining gap, and why it is `null`

Three values were originally recorded as gaps (`model.revision`, `sampling.top_p`,
`grpo.kl_coefficient`). Two were resolved in Phase 5.4; one remains.

**`model.revision` — resolved** to `2e1fd397ee46e1388853d2af2c993145b0f1098a`, the commit sha of
the cached snapshot. A Hugging Face repo name resolves to whatever is current, so two runs a
week apart could load different weights while every manifest field stayed identical — the most
expensive kind of irreproducibility, because it is invisible. The sha removes that.

**`sampling.top_p` — resolved** to `1.0` in **both** arms: `configs/eval/baseline.yaml`
(`sampling.top_p: 1.0`) and `configs/training/grpo_nemo_rl_a5000.yaml`
(`policy.generation.top_p: 1.0`). The trainer path is `policy.generation.top_p`, *not*
`policy.generation.vllm_cfg.top_p`: at NeMo RL `55296257`,
`VllmSpecificArgs` does not declare `top_p` at all. A sampler difference between the arms is
precisely the confound this file exists to prevent, so
`test_sampling_top_p_is_identical_in_both_arms` asserts all three agree.

**`grpo.kl_coefficient: null` — still a gap.** `0.0` would be a different and stronger claim:
that a KL term exists and is switched off. `docs/PHASE4_GRPO_AUDIT.md` found no reference policy
in the extracted GRPO path, so there is no KL term at all. `null` means *absent*;
`test_kl_absence_is_consistent_with_the_phase_4_audit` states this explicitly.

### 2.6 Reward, GRPO and data

| Setting | Value | Provenance | Why this value |
| --- | --- | --- | --- |
| `reward.live` | `reward_v1` | GROUNDED | the only reward any live path imports |
| `reward.analysed` | `reward_v2` | GROUNDED | computed offline by `scripts/reward_distribution.py` |
| `reward.range` | `[-1.0, 1.0]` | GROUNDED | the gates return `-1.0`; a clamp in the trainer collapses them (§Phase 0 of the runbook) |
| `grpo.advantage_estimator` | `group_relative` | GROUNDED | `training/grpo_math.py::group_advantages` |
| `grpo.clip_epsilon` | `0.28` | GROUNDED | `grpo_unsloth_a5000.yaml:111` `epsilon_high`; `0.2`/`0.28` is the GRPO convention |
| `grpo.drop_truncated_completions` | `true` | CHOSEN | a truncated completion is not a negative sample; scoring it as one trains on a harness limit. Named `drop_` rather than the trainer's `mask_truncated_completions` so this file is not mistaken for a TRL config by `rewards/tests/test_trainer_config_safety.py` |
| `data.split` | `train` | GROUNDED | training data |
| `data.validation_from` | `train` | CHOSEN | the held-out split is sealed and unavailable for validation |
| `data.heldout_enabled` | `false` | GROUNDED | `docs/heldout_seal.md` |

### 2.7 Hardware

Two distinct facts, recorded separately. Conflating them is what produced a wrong claim in an
earlier revision of this table, which listed the A5000 as the target without noting that the
machine in use had never been an A5000.

| Setting | Value | Provenance |
| --- | --- | --- |
| `target_gpu` | `RTX A6000 (48 GB)` | GROUNDED — `nvidia-smi` on the execution host, 2026-11-04: two `NVIDIA RTX A6000` at 49140 MiB, driver 535.309.01 |
| `historical_design_target_gpu` | `RTX A5000 (24 GB)` | HISTORICAL — the sizing target the protocol was reasoned against |
| `cuda_visible_devices` | `"0"` | CHOSEN — see [PHASE5_GPU_RESOURCE_RULES.md](PHASE5_GPU_RESOURCE_RULES.md) §2 for why it is set rather than assumed |

**The conservative settings were not loosened when the hardware was corrected.** Group size
stays 4, `max_seq_length` stays 8192, `gpu_memory_utilization` stays 0.4, and concurrency stays
1. The reason is stability and comparability rather than a belief that the card has 24 GB: every
VRAM figure and timing already in this project's record was produced under those settings, and a
run that raised them at the same time as everything else would confound its own result. The
24 GB column elsewhere in this document is therefore best read as "what was sized against",
not "what is installed".

The completed controlled run F4 also recorded an A6000
([PHASE_F4_CONTROLLED_UPDATE.md](PHASE_F4_CONTROLLED_UPDATE.md)), so this is a correction of a
documentation error rather than a change of hardware.

---

## 3. Phase 5.2 — the run manifest

`scripts/run_manifest.py` writes `manifest.json` into a run directory. It is written **before**
the run, because a manifest records *identity*, not *outcome*; outcome fields start null and are
filled by `--finish`.

```bash
# declare a run
uv run python scripts/run_manifest.py \
    --arm training --run-id t1_grpo_seed42 --split train --seed 42 \
    --input-checkpoint Qwen/Qwen2.5-Coder-1.5B-Instruct \
    -o results/raw/t1_grpo_seed42

# close it, on success or failure
uv run python scripts/run_manifest.py --finish completed -o results/raw/t1_grpo_seed42
uv run python scripts/run_manifest.py --finish failed "CUDA OOM at step 3" -o results/raw/t1_grpo_seed42
```

### 3.1 The fourteen required fields

| 5.2 item | Manifest field | Notes |
| --- | --- | --- |
| run id | `run_id` | supplied, not derived |
| date | `written_at` | UTC ISO-8601 |
| git commit | `git.commit` (+ `branch`, `dirty`) | a dirty tree is recorded as a gap |
| model | `protocol.model.name` (+ `revision`) | |
| dataset version | `dataset.version.sha256` | covers both splits |
| reward version | `reward.live` (+ `analysed`, `range`, both source hashes) | |
| trajectory schema version | `trajectory_schema.sha256` | `analysis/trace_schema.json` |
| training config hash | `arm_config.sha256` | hash of the arm's config |
| evaluation config hash | `protocol.file.sha256` | hash of the protocol |
| input checkpoint | `checkpoints.input` | `null` only when no checkpoint is loaded |
| output checkpoint | `checkpoints.output` | |
| GPU | `gpu` | `{present, devices, cuda_visible_devices, reason}` |
| seed | `seed` | |
| result folder | `artifacts.run_dir` | plus `trace`, `log`, `server_pid` |

Plus `outcome: {status, failure_reason}`, `timing: {started_at, finished_at, duration_s}`, and
`unresolved: [...]`. All fourteen are asserted present and resolved by
`test_manifest_has_every_field_phase_5_2_requires`.

### 3.2 The three guarantees

**An existing manifest is never silently replaced.** `write_manifest` raises `FileExistsError`
unless `--force`, and `--force` *renames* the old file to `manifest.superseded-<UTC>.json`
rather than deleting it. That the earlier manifest was wrong is information about the run. See
`test_a_differing_manifest_is_refused_without_force` and
`test_force_preserves_the_superseded_manifest_rather_than_deleting_it`.

**Re-running is safe.** A byte-identical manifest is not rewritten (`test_a_second_identical_manifest_is_not_rewritten`), so a
Makefile target can call the script every time. `outcome` and `timing` are excluded from the
comparison, so finishing a manifest does not make it look like a different run.

**Gaps are listed, not hidden.** `unresolved` collects every null, missing file, and dirty
tree. On this machine it is non-empty and says so. For the pilot arm, verified 2026-11-04:

```
unresolved (3): protocol.grpo.kl_coefficient,
                pilot.tasks (not yet selected),
                git.dirty (tree had uncommitted changes)
```

`protocol.model.revision` and `protocol.sampling.top_p` are no longer listed: both were resolved
in Phase 5.4, which is exactly the behaviour the field is for — a gap disappears when, and only
when, the value is actually decided. `git.dirty` clears once the Phase 5.4 work is committed.

**The seal is enforced at declaration.** `--arm training --split heldout` exits 2 and writes
nothing, so a run that intended to train on the held-out split cannot even be declared
(`test_cli_refuses_to_manifest_training_on_the_heldout_split`).

---

## 4. Phase 5.3 — GPU resource rules

See [PHASE5_GPU_RESOURCE_RULES.md](PHASE5_GPU_RESOURCE_RULES.md). The short version: GPU 0 only,
`CUDA_VISIBLE_DEVICES=0` on every command, never touch GPU 1 or another user's process, stop vLLM
after rollouts, free the card after training, check `nvidia-smi` before leaving, keep no idle
server, record the server PID and log path, and put a timeout on every long command.

The one non-obvious rule is that `CUDA_VISIBLE_DEVICES` **remaps** rather than filters, so an
unset variable makes `cuda:0` mean "physical GPU 0" only by luck of ordering.

---

## 5. What this phase does *not* establish

Following the rule that a claim needs a command behind it:

- **No run has used this protocol.** Every config carries `status: NEVER_EXECUTED`. The
  protocol is internally consistent and test-enforced; it is not validated on hardware.
- **The model revision is identified but unused.** `model.revision` pins the commit sha of the
  cached snapshot; that pin has not been exercised by a load.
- **`top_p` is now configured in both arms** (resolved 2026-11-04, `sampling.top_p: 1.0`);
  the remaining sampler confound risk is that the trainer's runtime acceptance of the key is
  unproven (§6).
- **Five iterations is a smoke, not a result.** A score from it says the pipeline works, not
  that the model improved. Per `AGENT_BUILD_SPEC.md` §1.2, a parameter change is not a model
  improvement.
- **The GPU rules were never executed.** No `nvidia-smi` output from the training box appears
  here, because none was produced.
- **The parity tests check declarations, not behaviour.** They prove the configs agree with each
  other and with the protocol; they cannot prove the trainer honours them. That requires the
  trainer-side inspection in `docs/RUN_ON_A5000.md` Phase 0.

---

## 6. Readiness, stated as two different questions

These are separated because conflating them is the failure this section exists to prevent: a
rollout-only pilot needs no trainer, so "the pilot is config-ready" says nothing about whether
training can run, and "training is blocked" says nothing about whether the pilot can.

### 6.1 Rollout-only pilot — config-ready

`configs/training/pilot_rollouts_1p5b.yaml`. This run samples the base model, scores with
credit, and writes traces. It performs **no parameter update** and does not depend on NeMo RL
being installed, parsed, or importable.

| # | Item | Value | Where |
| --- | --- | --- | --- |
| 1 | Model revision pinned | `2e1fd397ee46e1388853d2af2c993145b0f1098a` | `model.revision` |
| 2 | Temperature explicit | `1.0` | inherited from protocol; asserted by parity tests |
| 3 | Top-p explicit | `1.0` | protocol + baseline, both arms |
| 4 | Sequence budget explicit | `8192` | `model.max_seq_length` |
| 5 | Optimizer disabled | `optimizer_enabled: false` | asserted by test |
| 6 | Training iterations zero | `training_iterations: 0` | asserted by test |
| 7 | Held-out disabled | `heldout_enabled: false` | asserted by test |
| 8 | `reward_v1` live | `reward.live: reward_v1` | `rewards/compose.py` is the live path |
| 9 | `reward_v2` offline only | not imported by any live path | `rewards/v2.py` |
| 10 | Training task selected | **`tasks: null`** — deliberate, see below | `pilot_rollouts_1p5b.yaml` |

**Item 10 is an honest gap, not a satisfied one.** The pilot does not name a task because
inventing a task ID would be inventing the thing the run is supposed to measure. The pilot is
therefore *config-complete* and *not launch-ready* until a task is named; that is a one-value
decision, not a code change.

The pilot needs **no** NeMo RL field at all: it does not construct a `PolicyConfig`, so the
`policy.max_total_sequence_length` / `max_model_len` mapping below is not its concern.

### 6.2 Formal NeMo RL training — still BLOCKED

`configs/training/grpo_nemo_rl_a5000.yaml`. Every key below now has a cited source, and none
of it has been executed. **This document does not claim formal training readiness.**

The blockers, each independently sufficient:

1. **Pinned NeMo RL is not available in a compatible environment.** It requires
   `>=3.13.13,<3.14`; this workspace runs 3.12 and NeMo RL is not installed
   (`import nemo_rl` → `ModuleNotFoundError`).
2. **The actual config loader has never parsed the trainer YAML.** `policy` is a raw
   `TypedDict` inside `MasterConfig(extra="allow")`, so a misplaced key is *not* rejected at
   load — and the runtime consumers use direct subscripts, so a missing one is a `KeyError` at
   first use rather than a validation error at startup. Only a real parse settles this.
3. **Runtime sampling settings are unconfirmed.** `temperature` / `top_p` at
   `policy.generation.*` are source-grounded; that the loader accepts them is not observed.
4. **Runtime sequence-length settings are unconfirmed.** `max_total_sequence_length: 8192` and
   `vllm_cfg.max_model_len: 8192` are source-grounded; that a run at 8192 does not OOM on the
   target card is not observed.
5. **No parser-only or config-only smoke test has passed**, because it cannot be run here.

The sequence-length mapping itself is settled (§6.3). What is *not* settled is that the pinned
runtime accepts it — a different claim, and the remaining blocker.

### 6.3 The sequence-length mapping, as applied

| Concept | Field | Value | Limits |
| --- | --- | --- | --- |
| Protocol budget | `model.max_seq_length` | 8192 | the arm-independent number |
| **Total** prompt + completion | `policy.max_total_sequence_length` | 8192 | a whole trajectory |
| **Engine** context capacity | `policy.generation.vllm_cfg.max_model_len` | 8192 | what vLLM can hold |
| **Completion** only | `policy.generation.max_new_tokens` | **unset** | not defined by the protocol |

`max_new_tokens` is unset on purpose: it caps generation only, the protocol defines no
completion budget, and copying `max_total_sequence_length` into it would be inventing a
protocol value. The three are not interchangeable.

---

## 9. Change control

Once a run has been recorded against this protocol, the way to change a value is:

1. change `configs/training/protocol_1p5b.yaml` (or the arm config);
2. note that the config hash changes, so new manifests cannot be compared with old ones — this
   is the mechanism, not a problem;
3. start new run ids; never re-use a run id across a protocol change;
4. if the change was a correction to something wrong, say so in the new manifest's `--note`.

Retro-fitting a value to match a result already obtained would make the protocol a description
of what happened rather than a control on what may happen, which is the failure mode Phase 5 was
written to prevent.
