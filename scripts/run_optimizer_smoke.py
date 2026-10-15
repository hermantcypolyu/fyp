"""One-step GRPO-style optimizer smoke for multi-turn coding rollouts.

This script is deliberately small and trainer-independent. It proves that:
- real multi-turn rollout token IDs and old log-probabilities are usable;
- group-relative advantages reach a masked policy loss;
- tool outputs do not receive gradients;
- LoRA parameters update;
- the adapter checkpoint saves and reloads correctly.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.rollout_adapter import EpisodeTrainingSample, adapt_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
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
    parser.add_argument(
        "--adapter-checkpoint",
        type=Path,
        default=None,
        help=(
            "Existing LoRA checkpoint to continue training from. "
            "If omitted, a new LoRA adapter is created."
        ),
    )
    return parser.parse_args()


def load_samples(path: Path) -> list[EpisodeTrainingSample]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    samples: list[EpisodeTrainingSample] = []

    for index, row in enumerate(rows, start=1):
        sample = adapt_rollout(row)

        if sample is None:
            print(f"Discarding rollout {index}: no trainable turns")
            continue

        samples.append(sample)

    if len(samples) < 2:
        raise RuntimeError(
            "Optimizer smoke requires at least two scored rollouts."
        )

    task_ids = {sample.task_id for sample in samples}

    if len(task_ids) != 1:
        raise RuntimeError(
            "All GRPO group members must come from the same task. "
            f"Found: {sorted(task_ids)}"
        )

    return samples


def compute_advantages(
    samples: list[EpisodeTrainingSample],
) -> tuple[list[float], float, float]:
    rewards = [sample.reward for sample in samples]
    reward_mean = statistics.fmean(rewards)
    reward_std = statistics.pstdev(rewards)

    if reward_std <= 0:
        raise RuntimeError(
            "Reward standard deviation is zero. "
            "This group has no learning signal."
        )

    advantages = [
        (reward - reward_mean) / (reward_std + 1.0e-6)
        for reward in rewards
    ]

    if not all(math.isfinite(value) for value in advantages):
        raise RuntimeError("Advantages contain a non-finite value.")

    if advantages[rewards.index(min(rewards))] != min(advantages):
        raise RuntimeError(
            "The lowest reward did not receive the lowest advantage."
        )

    return advantages, reward_mean, reward_std


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
    full_token_ids = (
        list(prompt_token_ids)
        + list(generation_token_ids)
    )

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
        raise RuntimeError(
            "Current-logprob extraction produced the wrong number "
            "of generation positions."
        )

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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoint"
    metrics_path = args.output_dir / "training_metrics.json"
    advantages_path = args.output_dir / "advantages.json"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda:0")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(device))
    print("Loading rollouts:", args.rollouts)

    samples = load_samples(args.rollouts)
    advantages, reward_mean, reward_std = compute_advantages(samples)

    rewards = [sample.reward for sample in samples]

    advantage_records = [
        {
            "rollout_index": index,
            "task_id": sample.task_id,
            "reward": sample.reward,
            "advantage": advantage,
            "trainable_turns": len(sample.turns),
            "trainable_tokens": sum(
                len(turn.generation_token_ids)
                for turn in sample.turns
            ),
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

    total_trainable_turns = sum(
        len(sample.turns)
        for sample in samples
    )
    total_trainable_tokens = sum(
        len(turn.generation_token_ids)
        for sample in samples
        for turn in sample.turns
    )

    print("Rollouts:", len(samples))
    print("Rewards:", rewards)
    print("Advantages:", advantages)
    print("Trainable turns:", total_trainable_turns)
    print("Trainable tokens:", total_trainable_tokens)

    if total_trainable_tokens <= 0:
        raise RuntimeError("No trainable tokens were found.")

    torch.cuda.reset_peak_memory_stats(device)

    load_started = time.monotonic()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
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

    if args.adapter_checkpoint is not None:
        if not args.adapter_checkpoint.is_dir():
            raise FileNotFoundError(
                "Adapter checkpoint directory does not exist: "
                f"{args.adapter_checkpoint}"
            )

        print(
            "Loading trainable LoRA checkpoint:",
            args.adapter_checkpoint,
        )

        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_checkpoint,
            is_trainable=True,
        )
    else:
        print("Creating a new LoRA adapter")
        model = get_peft_model(base_model, lora_config)

    model.train()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError("LoRA created no trainable parameters.")

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    print("Trainable LoRA parameters:", trainable_parameter_count)
    print(
        "Model load seconds:",
        round(time.monotonic() - load_started, 3),
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    before = trainable_parameter_snapshot(model)

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
                    f"Rollout {rollout_index}, output "
                    f"{turn.output_index}: current/old logprob "
                    "shapes do not match."
                )

            if not torch.isfinite(current_log_probs).all():
                raise RuntimeError("Current logprobs are non-finite.")

            if not torch.isfinite(old_log_probs).all():
                raise RuntimeError("Old logprobs are non-finite.")

            log_ratio = current_log_probs - old_log_probs

            # Protect the smoke run from overflow while preserving the
            # PPO/GRPO clipping calculation.
            ratio = torch.exp(
                torch.clamp(log_ratio, min=-20.0, max=20.0)
            )

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

            ratio_values.extend(
                ratio.detach().float().cpu().tolist()
            )
            current_logprob_values.extend(
                current_log_probs.detach().float().cpu().tolist()
            )
            old_logprob_values.extend(
                old_log_probs.detach().float().cpu().tolist()
            )

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

        raw_gradient_norm += float(
            parameter.grad.detach().float().square().sum().item()
        )

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

    after = trainable_parameter_snapshot(model)

    changed_parameters, delta_l2, max_absolute_delta = parameter_delta(
        before,
        after,
    )

    if changed_parameters <= 0:
        raise RuntimeError(
            "optimizer.step() completed but no LoRA parameter changed."
        )

    if delta_l2 <= 0:
        raise RuntimeError(
            "LoRA parameter delta is zero after optimizer.step()."
        )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    saved_adapter = adapter_parameter_snapshot(model)

    peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    training_seconds = time.monotonic() - training_started

    metrics: dict[str, Any] = {
        "model": args.model,
        "task_id": samples[0].task_id,
        "rollouts": len(samples),
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
        "clip_epsilon": args.clip_epsilon,
        "lora_rank": args.lora_rank,
        "trainable_parameter_count": trainable_parameter_count,
        "policy_loss": total_loss_value,
        "loss_is_finite": math.isfinite(total_loss_value),
        "raw_gradient_norm": raw_gradient_norm,
        "clipped_gradient_norm": clipped_gradient_norm_value,
        "gradient_norm_is_finite": math.isfinite(
            clipped_gradient_norm_value
        ),
        "ratio_mean": statistics.fmean(ratio_values),
        "ratio_min": min(ratio_values),
        "ratio_max": max(ratio_values),
        "old_logprob_mean": statistics.fmean(
            old_logprob_values
        ),
        "current_logprob_mean": statistics.fmean(
            current_logprob_values
        ),
        "changed_lora_parameters": changed_parameters,
        "parameter_delta_l2": delta_l2,
        "max_absolute_parameter_delta": max_absolute_delta,
        "optimizer_step_completed": True,
        "checkpoint_saved": checkpoint_dir.exists(),
        "checkpoint_reloaded": False,
        "peak_vram_mb": peak_vram_mb,
        "training_seconds": training_seconds,
        "gpu": torch.cuda.get_device_name(device),
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\n===== OPTIMIZER STEP =====")
    print("Policy loss:", total_loss_value)
    print("Raw gradient norm:", raw_gradient_norm)
    print("Clipped gradient norm:", clipped_gradient_norm_value)
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

    reload_base = AutoModelForCausalLM.from_pretrained(
        args.model,
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

    common_names = sorted(
        set(saved_adapter) & set(reloaded_adapter)
    )

    if not common_names:
        raise RuntimeError(
            "No common LoRA parameter names were found after reload."
        )

    reload_max_difference = max(
        float(
            (
                saved_adapter[name]
                - reloaded_adapter[name]
            ).abs().max().item()
        )
        for name in common_names
    )

    if reload_max_difference != 0.0:
        raise RuntimeError(
            "Reloaded LoRA parameters do not exactly match "
            f"the saved checkpoint. max_difference={reload_max_difference}"
        )

    metrics["checkpoint_reloaded"] = True
    metrics["reloaded_lora_parameter_count"] = len(common_names)
    metrics["reload_max_parameter_difference"] = reload_max_difference

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("Reloaded LoRA tensors:", len(common_names))
    print("Reload max difference:", reload_max_difference)
    print("\nPASS: one-step optimizer smoke completed")


if __name__ == "__main__":
    main()
