"""Phase 7 one-step GRPO + LoRA feasibility smoke for a 3B coding model.

This script is deliberately small and trainer-independent. It proves that:
- real multi-turn rollout token IDs and old log-probabilities are usable;
- group-relative advantages reach a masked policy loss;
- tool outputs do not receive gradients;
- LoRA parameters update;
- the base-model weights are byte-identical afterwards;
- the adapter checkpoint saves and reloads correctly.

Phase 7 scope, enforced by `training/phase7_guards.py`
------------------------------------------------------
Exactly one model (the pinned 3B revision), exactly one four-rollout group,
exactly one forward/backward/optimizer step, LoRA only, no generation of any
kind, no existing adapter. The guards refuse before any model or tokenizer load,
so a mistyped invocation costs nothing.

What this script does *not* claim
---------------------------------
There is no reference-policy KL term, so the KL column reads
`NOT_APPLICABLE_CURRENT_IMPLEMENTATION` rather than 0.0. No rollout is generated:
the input is recorded Phase 6M evidence. A single optimizer step on four rollouts
is a feasibility check, not a training result, and no performance claim follows
from it.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.grpo_math import DegenerateGroupError, group_advantages
from training.phase7_guards import (
    EXPECTED_OPTIMIZER_STEPS,
    KL_STATUS_NOT_APPLICABLE,
    PERMITTED_MODEL_NAME,
    PERMITTED_MODEL_REVISION,
    OptimizerStepCounter,
    Phase7ExecutionPlan,
    Phase7Violation,
    assert_lora_only_trainable,
    assert_phase7_model_is_permitted,
    base_tensor_manifest,
    compare_base_manifests,
    count_changed_tensors,
    load_permitted_model,
    load_permitted_tokenizer,
    normalise_for_guard,
    tensor_digest,
    validate_execution_plan,
    validate_rollout_group,
)
from training.rollout_adapter import EpisodeTrainingSample, adapt_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--rollouts",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    # The default is the permitted model rather than the 1.5B, so the bare
    # invocation is the correct one. ADR-0008 excludes the 1.5B from Phase 7.
    parser.add_argument(
        "--model",
        default=PERMITTED_MODEL_NAME,
    )
    # Pinned explicitly rather than resolved from the cache. A cache-state
    # resolution is not a revision, and Phase 6M ran this exact snapshot.
    parser.add_argument(
        "--revision",
        default=PERMITTED_MODEL_REVISION,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.2,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-5,
    )
    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
    )
    # `--adapter-checkpoint` used to allow continuing an existing adapter. It is
    # gone rather than defaulted to None: an option that exists can be passed, and
    # Phase 7 must start from a fresh adapter. Any invocation using it now fails at
    # argument parsing, before any guard or GPU work.
    parser.add_argument(
        "--diagnostic-input-manifest",
        type=Path,
        default=None,
        help=(
            "Phase 7 Part C input manifest to verify against these rollouts before "
            "training. Optional, but recorded in the metrics when supplied."
        ),
    )
    return parser.parse_args()


def load_samples(path: Path) -> tuple[list[EpisodeTrainingSample], list[dict[str, Any]]]:
    """Adapt the recorded rollout rows, refusing anything but the Phase 7 group.

    Returns the adapted samples *and* the raw rows, because the guard needs the
    raw `rollout_index` / `reward` / `task_id` fields that the adapted dataclass
    does not carry, and re-reading the file to get them would risk reading a
    different file than the one that was validated.

    A row that adapts to ``None`` is a hard error here. `adapt_rollout` returns
    ``None`` for "no trainable turns", and silently dropping such a row would leave
    a three-member group whose advantages are computed over three rewards. The old
    behaviour printed a warning and continued; Phase 7 requires four rows or none.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    # Cardinality, single-task, uniqueness and the exact reward sequence are all
    # checked on the raw rows before a single model or tokenizer is constructed.
    # The recorded trace nests the guard's fields (`agent.rollout_index`,
    # `result.reward`, `task.id`), so project first; the guard validates the
    # projection while the rows below stay untouched.
    validate_rollout_group(normalise_for_guard(rows))

    samples: list[EpisodeTrainingSample] = []

    for index, row in enumerate(rows, start=1):
        sample = adapt_rollout(row)

        if sample is None:
            raise Phase7Violation(
                f"Rollout {index} has no trainable turns. Dropping it would change the group "
                "from four members to three and recompute the advantages over a different group."
            )

        samples.append(sample)

    if len(samples) != len(rows):
        raise Phase7Violation(f"Expected {len(rows)} training samples; adapted {len(samples)}.")

    return samples, rows


def compute_advantages(
    samples: list[EpisodeTrainingSample],
) -> tuple[list[float], float, float]:
    """Group statistics for one GRPO group.

    Delegates to `training.grpo_math`, which is torch-free and unit-tested. The
    degenerate-group check used to live here, where no test could reach it.
    """
    rewards = [sample.reward for sample in samples]

    try:
        return group_advantages(rewards)
    except DegenerateGroupError as error:
        raise RuntimeError(str(error)) from error


def trainable_parameter_snapshot(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def adapter_parameter_snapshot(
    model: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }


def parameter_delta(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> tuple[int, float, float]:
    if set(before) != set(after):
        raise RuntimeError("Trainable parameter names changed during the step.")

    changed = 0
    squared_delta = 0.0
    max_absolute_delta = 0.0

    for name in before:
        delta = after[name] - before[name]
        absolute_max = float(delta.abs().max().item())
        squared_delta += float(delta.square().sum().item())
        max_absolute_delta = max(max_absolute_delta, absolute_max)

        if absolute_max > 0:
            changed += 1

    return changed, math.sqrt(squared_delta), max_absolute_delta


def current_generation_log_probs(
    model: torch.nn.Module,
    *,
    prompt_token_ids: tuple[int, ...],
    generation_token_ids: tuple[int, ...],
    temperature: float,
    device: torch.device,
) -> torch.Tensor:
    full_token_ids = list(prompt_token_ids) + list(generation_token_ids)

    input_ids = torch.tensor(
        [full_token_ids],
        dtype=torch.long,
        device=device,
    )

    outputs = model(
        input_ids=input_ids,
        use_cache=False,
    )

    prompt_length = len(prompt_token_ids)
    generation_length = len(generation_token_ids)

    # Token at sequence position n is predicted by logits at n - 1.
    generation_logits = outputs.logits[
        0,
        prompt_length - 1 : prompt_length + generation_length - 1,
        :,
    ]

    if generation_logits.shape[0] != generation_length:
        raise RuntimeError("Current-logprob extraction produced the wrong number of generation positions.")

    generation_targets = input_ids[
        0,
        prompt_length : prompt_length + generation_length,
    ]

    log_probabilities = torch.log_softmax(
        generation_logits.float() / temperature,
        dim=-1,
    )

    selected = log_probabilities.gather(
        dim=-1,
        index=generation_targets.unsqueeze(-1),
    ).squeeze(-1)

    return selected


def main() -> None:
    args = parse_args()

    if args.temperature <= 0:
        raise ValueError("temperature must be positive")

    # ---- Phase 7 environment gate: interpreter and device visibility ----
    #
    # Both checks run before the model guard and before any CUDA initialisation.
    # The wrong interpreter is the failure mode that is easiest to miss: a bare
    # `python3` on this host is 3.10 and cannot even import this module, while a
    # third-party venv may carry a different torch. Pinning the interpreter here
    # means a mistaken invocation stops immediately instead of training for
    # several minutes on an unintended stack.
    if sys.version_info[:2] != (3, 12):
        raise Phase7Violation(
            f"Python 3.12.x is required for Phase 7; this interpreter is "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} "
            f"({sys.executable})."
        )

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print("Interpreter:", sys.executable)
    print("Python:", sys.version.split()[0])
    print("CUDA_VISIBLE_DEVICES:", visible_devices or "<unset>")

    # Exactly one visible device, and it must be requested explicitly. Without
    # this, an unset CUDA_VISIBLE_DEVICES would let torch enumerate every card and
    # the `cuda:0` below could silently land on someone else's GPU.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device_count = torch.cuda.device_count()
    print("torch.cuda.device_count():", device_count)
    if device_count != 1:
        raise Phase7Violation(
            f"Phase 7 requires exactly one visible CUDA device; torch sees {device_count}. "
            "Set CUDA_VISIBLE_DEVICES to a single GPU."
        )

    # ---- Phase 7 gate: everything here happens before any model load ----
    #
    # Order matters and is deliberate. The model guard runs first because it is
    # the one prohibition with no override, and because a refused 7B must not even
    # initialise CUDA. `assert_phase7_model_is_permitted` takes `torch.cuda.is_initialized`
    # as a callback so the ordering is checked rather than assumed.
    assert_phase7_model_is_permitted(
        args.model,
        args.revision,
        cuda_initialised=torch.cuda.is_initialized,
    )

    validate_execution_plan(
        Phase7ExecutionPlan(
            model_name=args.model,
            model_revision=args.revision,
        )
    )

    # One fresh directory per run. Refusing to reuse it is cheaper than discovering
    # afterwards that the checkpoint on disk was the previous run's.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise Phase7Violation(
            f"Output directory is not empty: {args.output_dir}. Phase 7 writes to a fresh "
            "directory so a stale checkpoint can never be mistaken for this run's."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint"
    metrics_path = args.output_dir / "training_metrics.json"
    advantages_path = args.output_dir / "advantages.json"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda:0")

    # Logical index 0 is the only visible device, so it is the only device that
    # can be reached. Record what it physically maps to, because the record is
    # the evidence that this run did not land on another card.
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    used_bytes = total_bytes - free_bytes
    mib = 1024 * 1024
    physical_indices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print("GPU name:", torch.cuda.get_device_name(device))
    print("Visible logical index: 0 (physical CUDA_VISIBLE_DEVICES entry:", physical_indices, ")")
    print(
        f"Memory before load: total {total_bytes // mib} MiB, used {used_bytes // mib} MiB, free {free_bytes // mib} MiB"
    )
    print("Model:", args.model)
    print("Revision:", args.revision)
    print("Loading rollouts:", args.rollouts)

    samples, raw_rows = load_samples(args.rollouts)
    advantages, reward_mean, reward_std = compute_advantages(samples)

    rewards = [sample.reward for sample in samples]

    advantage_records = [
        {
            "rollout_index": index,
            "task_id": sample.task_id,
            "reward": sample.reward,
            "advantage": advantage,
            "trainable_turns": len(sample.turns),
            "trainable_tokens": sum(len(turn.generation_token_ids) for turn in sample.turns),
        }
        for index, (sample, advantage) in enumerate(
            zip(samples, advantages, strict=True),
            start=1,
        )
    ]

    advantages_path.write_text(
        json.dumps(
            {
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "records": advantage_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    total_trainable_turns = sum(len(sample.turns) for sample in samples)
    total_trainable_tokens = sum(len(turn.generation_token_ids) for sample in samples for turn in sample.turns)

    print("Rollouts:", len(samples))
    print("Rewards:", rewards)
    print("Advantages:", advantages)
    print("Trainable turns:", total_trainable_turns)
    print("Trainable tokens:", total_trainable_tokens)

    if total_trainable_tokens <= 0:
        raise RuntimeError("No trainable tokens were found.")

    torch.cuda.reset_peak_memory_stats(device)

    load_started = time.monotonic()

    # `load_permitted_*` re-verifies the pin and passes `revision=` explicitly, so
    # the weights are the Phase 6M snapshot even if the local cache has moved on.
    tokenizer = load_permitted_tokenizer(
        AutoTokenizer.from_pretrained,
        model_name=args.model,
        revision=args.revision,
        local_files_only=True,
    )

    base_model = load_permitted_model(
        AutoModelForCausalLM.from_pretrained,
        model_name=args.model,
        revision=args.revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    base_model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # Phase 7 always builds a fresh adapter. The previous branch that loaded an
    # existing checkpoint with `is_trainable=True` is gone; `validate_execution_plan`
    # refuses that plan and the CLI no longer offers the flag.
    print("Creating a new LoRA adapter")
    model = get_peft_model(base_model, lora_config)

    model.train()

    # Every trainable parameter must be a LoRA tensor and no base tensor may be
    # trainable. Counts are measured here and reported, not asserted afterwards.
    trainable_parameter_count_named, frozen_parameter_count = assert_lora_only_trainable(model.named_parameters())

    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    if not trainable_parameters:
        raise RuntimeError("LoRA created no trainable parameters.")

    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)

    # ---- Deterministic base-model manifest, before any forward pass ----
    #
    # Digests of every non-LoRA tensor. Recomputed after the step and compared
    # tensor by tensor, which is what proves the base model stayed frozen rather
    # than merely asserting that PEFT usually keeps it that way.
    base_manifest_before = base_tensor_manifest(model.named_parameters())

    print("Trainable LoRA parameter tensors:", trainable_parameter_count_named)
    print("Frozen parameter tensors:", frozen_parameter_count)
    print("Trainable LoRA parameters:", trainable_parameter_count)
    print("Base tensors fingerprinted:", len(base_manifest_before))
    print(
        "Model load seconds:",
        round(time.monotonic() - load_started, 3),
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )

    before = trainable_parameter_snapshot(model)

    # LoRA digests before the step, for the "at least one adapter tensor moved" check.
    lora_digest_before = {
        name: tensor_digest(parameter) for name, parameter in model.named_parameters() if "lora_" in name
    }

    # A real counter, not a boolean. `assert_zero` here and `assert_exactly_one`
    # after the step are the enforcement of "exactly one update".
    step_counter = OptimizerStepCounter()
    step_counter.assert_zero()

    optimizer.zero_grad(set_to_none=True)

    total_loss_value = 0.0
    ratio_values: list[float] = []
    current_logprob_values: list[float] = []
    old_logprob_values: list[float] = []

    training_started = time.monotonic()

    for rollout_index, (sample, advantage) in enumerate(
        zip(samples, advantages, strict=True),
        start=1,
    ):
        advantage_tensor = torch.tensor(
            advantage,
            dtype=torch.float32,
            device=device,
        )

        for turn in sample.turns:
            current_log_probs = current_generation_log_probs(
                model,
                prompt_token_ids=turn.prompt_token_ids,
                generation_token_ids=turn.generation_token_ids,
                temperature=args.temperature,
                device=device,
            )

            old_log_probs = torch.tensor(
                turn.generation_log_probs,
                dtype=torch.float32,
                device=device,
            )

            if current_log_probs.shape != old_log_probs.shape:
                raise RuntimeError(
                    f"Rollout {rollout_index}, output {turn.output_index}: current/old logprob shapes do not match."
                )

            if not torch.isfinite(current_log_probs).all():
                raise RuntimeError("Current logprobs are non-finite.")

            if not torch.isfinite(old_log_probs).all():
                raise RuntimeError("Old logprobs are non-finite.")

            log_ratio = current_log_probs - old_log_probs

            # The clamp below exists to protect this run from overflow while
            # preserving the PPO/GRPO clipping calculation. The ratio,
            # clipping, surrogate and token-weighted loss are implemented here
            # with vectorised torch operations; training/grpo_math.py holds
            # torch-free scalar reference versions of the same formulas that
            # the CPU unit tests exercise. Only `group_advantages` is shared
            # code that this script actually calls.
            #
            # The scalar reference tests therefore assert the intended
            # mathematical behaviour, not that this vectorised implementation is
            # numerically identical to it. An equivalence test comparing scalar
            # reference outputs against these torch outputs on identical inputs
            # is not yet written and would require an environment with PyTorch.
            ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))

            clipped_ratio = torch.clamp(
                ratio,
                1.0 - args.clip_epsilon,
                1.0 + args.clip_epsilon,
            )

            unclipped_objective = ratio * advantage_tensor
            clipped_objective = clipped_ratio * advantage_tensor

            surrogate = torch.minimum(
                unclipped_objective,
                clipped_objective,
            )

            # Token-weighted group loss. Calling backward per turn releases
            # each long sequence graph before processing the next turn.
            turn_loss = -surrogate.sum() / total_trainable_tokens

            if not torch.isfinite(turn_loss):
                raise RuntimeError("Policy loss became non-finite.")

            turn_loss.backward()

            total_loss_value += float(turn_loss.detach().cpu().item())

            ratio_values.extend(ratio.detach().float().cpu().tolist())
            current_logprob_values.extend(current_log_probs.detach().float().cpu().tolist())
            old_logprob_values.extend(old_log_probs.detach().float().cpu().tolist())

            del current_log_probs
            del old_log_probs
            del ratio
            del clipped_ratio
            del surrogate
            del turn_loss

    raw_gradient_norm = 0.0

    for parameter in trainable_parameters:
        if parameter.grad is None:
            continue

        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("A LoRA gradient is non-finite.")

        raw_gradient_norm += float(parameter.grad.detach().float().square().sum().item())

    raw_gradient_norm = math.sqrt(raw_gradient_norm)

    if not math.isfinite(raw_gradient_norm):
        raise RuntimeError("Gradient norm is non-finite.")

    clipped_gradient_norm = clip_grad_norm_(
        trainable_parameters,
        max_norm=args.max_grad_norm,
    )

    clipped_gradient_norm_value = float(
        clipped_gradient_norm.detach().cpu().item()
        if isinstance(clipped_gradient_norm, torch.Tensor)
        else clipped_gradient_norm
    )

    if not math.isfinite(clipped_gradient_norm_value):
        raise RuntimeError("Clipped gradient norm is non-finite.")

    optimizer.step()

    # The counter only moves after `step()` returned normally, so a raise above
    # leaves it at zero and this assertion records a failure rather than a claim.
    step_counter.record_completed_step()
    step_counter.assert_exactly_one()

    after = trainable_parameter_snapshot(model)

    changed_parameters, delta_l2, max_absolute_delta = parameter_delta(
        before,
        after,
    )

    if changed_parameters <= 0:
        raise RuntimeError("optimizer.step() completed but no LoRA parameter changed.")

    if delta_l2 <= 0:
        raise RuntimeError("LoRA parameter delta is zero after optimizer.step().")

    # ---- LoRA verification: at least one adapter tensor must have moved ----
    lora_digest_after = {
        name: tensor_digest(parameter) for name, parameter in model.named_parameters() if "lora_" in name
    }

    changed_lora_tensors = count_changed_tensors(lora_digest_before, lora_digest_after)

    if not changed_lora_tensors:
        raise RuntimeError("No LoRA tensor changed content after optimizer.step().")

    # ---- Base-model freeze verification: zero tensors may have changed ----
    #
    # Recomputed from the live model, not from a saved copy, so this measures the
    # weights that would actually be saved.
    base_manifest_after = base_tensor_manifest(model.named_parameters())
    base_changes = compare_base_manifests(base_manifest_before, base_manifest_after)

    total_base_changes = sum(len(names) for names in base_changes.values())

    if total_base_changes != 0:
        raise RuntimeError(
            "Base-model tensors changed during a LoRA-only step: "
            + json.dumps({kind: names[:10] for kind, names in base_changes.items() if names})
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    saved_adapter = adapter_parameter_snapshot(model)

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    training_seconds = time.monotonic() - training_started

    metrics: dict[str, Any] = {
        "phase": 7,
        "model": args.model,
        "model_revision": args.revision,
        # The KL column is a string because the implementation has no reference
        # policy and therefore no KL to report. Writing 0.0 here would claim a
        # measurement that was never taken.
        "reference_policy_kl": KL_STATUS_NOT_APPLICABLE,
        "task_id": samples[0].task_id,
        "rollouts": len(samples),
        "rollout_indices": [row.get("rollout_index") for row in raw_rows],
        "rewards": rewards,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "advantages": advantages,
        "advantage_mean": statistics.fmean(advantages),
        "advantage_std": statistics.pstdev(advantages),
        "trainable_turns": total_trainable_turns,
        "trainable_tokens": total_trainable_tokens,
        "temperature": args.temperature,
        "learning_rate": args.learning_rate,
        "adamw_betas": [0.9, 0.999],
        "adamw_eps": 1.0e-8,
        "weight_decay": 0.0,
        "clip_epsilon": args.clip_epsilon,
        "max_grad_norm": args.max_grad_norm,
        "gradient_clipping_bound": clipped_gradient_norm_value < raw_gradient_norm,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_rank * 2,
        "iterations": 1,
        "planned_optimizer_steps": EXPECTED_OPTIMIZER_STEPS,
        # Measured, not asserted: the counter increments only after step() returns.
        "optimizer_step_count": step_counter.count,
        "optimizer_step_completed": step_counter.count == EXPECTED_OPTIMIZER_STEPS,
        "trainable_parameter_count": trainable_parameter_count,
        "trainable_parameter_tensors": trainable_parameter_count_named,
        "frozen_parameter_tensors": frozen_parameter_count,
        "policy_loss": total_loss_value,
        "loss_is_finite": math.isfinite(total_loss_value),
        "raw_gradient_norm": raw_gradient_norm,
        "clipped_gradient_norm": clipped_gradient_norm_value,
        "gradient_norm_is_finite": math.isfinite(clipped_gradient_norm_value),
        "ratio_mean": statistics.fmean(ratio_values),
        "ratio_min": min(ratio_values),
        "ratio_max": max(ratio_values),
        "old_logprob_mean": statistics.fmean(old_logprob_values),
        "current_logprob_mean": statistics.fmean(current_logprob_values),
        "changed_lora_parameters": changed_parameters,
        "changed_lora_tensors": len(changed_lora_tensors),
        "changed_lora_tensor_names": changed_lora_tensors[:20],
        "parameter_delta_l2": delta_l2,
        "max_absolute_parameter_delta": max_absolute_delta,
        "base_tensors_fingerprinted": len(base_manifest_before),
        "base_tensors_changed": total_base_changes,
        "base_tensor_changes": {kind: names for kind, names in base_changes.items() if names},
        "checkpoint_saved": checkpoint_dir.exists(),
        "checkpoint_reloaded": False,
        "prohibited_paths_disabled": {
            "rollout_generation": False,
            "evaluation_generation": False,
            "heldout_evaluation": False,
            "existing_adapter": False,
            "multi_iteration": False,
        },
        "peak_vram_mb": peak_vram_mb,
        "training_seconds": training_seconds,
        "gpu": torch.cuda.get_device_name(device),
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n===== OPTIMIZER STEP =====")
    print("Optimizer steps:", step_counter.count)
    print("Policy loss:", total_loss_value)
    print("Raw gradient norm:", raw_gradient_norm)
    print("Clipped gradient norm:", clipped_gradient_norm_value)
    print("Changed LoRA tensors:", len(changed_lora_tensors))
    print("Changed base tensors:", total_base_changes)
    print("Changed LoRA parameters:", changed_parameters)
    print("Parameter delta L2:", delta_l2)
    print("Max absolute delta:", max_absolute_delta)
    print("Peak VRAM MiB:", round(peak_vram_mb, 2))
    print("Checkpoint:", checkpoint_dir)

    del optimizer
    del model
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\nReloading checkpoint...")

    reload_base = load_permitted_model(
        AutoModelForCausalLM.from_pretrained,
        model_name=args.model,
        revision=args.revision,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    reloaded_model = PeftModel.from_pretrained(
        reload_base,
        checkpoint_dir,
        is_trainable=False,
    )

    reloaded_adapter = adapter_parameter_snapshot(reloaded_model)

    # Strict key equality, not an intersection. The previous version compared only
    # keys present on both sides, so a checkpoint missing an adapter tensor, or
    # carrying an extra one, still passed.
    missing_keys = sorted(set(saved_adapter) - set(reloaded_adapter))
    unexpected_keys = sorted(set(reloaded_adapter) - set(saved_adapter))

    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Reloaded adapter keys differ from the saved adapter. "
            f"missing={missing_keys[:10]} unexpected={unexpected_keys[:10]}"
        )

    reload_max_difference = max(
        float((saved_adapter[name] - reloaded_adapter[name]).abs().max().item()) for name in sorted(saved_adapter)
    )

    if reload_max_difference != 0.0:
        raise RuntimeError(
            "Reloaded LoRA parameters do not exactly match "
            f"the saved checkpoint. max_difference={reload_max_difference}"
        )

    metrics["checkpoint_reloaded"] = True
    metrics["reloaded_lora_parameter_count"] = len(saved_adapter)
    metrics["reload_missing_adapter_keys"] = missing_keys
    metrics["reload_unexpected_adapter_keys"] = unexpected_keys
    metrics["reload_max_parameter_difference"] = reload_max_difference

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("Reloaded LoRA tensors:", len(saved_adapter))
    print("Reload max difference:", reload_max_difference)
    print("\nPASS: one-step optimizer smoke completed")


if __name__ == "__main__":
    main()
