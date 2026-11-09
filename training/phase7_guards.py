"""Phase 7 fail-closed guards and deterministic integrity checks.

Why this module is separate from `scripts/run_optimizer_smoke.py`
-----------------------------------------------------------------
The Phase 7 feasibility smoke is allowed to do exactly one thing: one forward
pass, one backward pass and one optimizer step on one four-rollout GRPO group,
with a LoRA adapter. Everything else is a defect. "Everything else" is far
cheaper to describe than to test, so the refusals live here, in a module that
imports no torch, no transformers and no peft, and therefore can be exercised on
a CPU-only machine during `make check`.

The script keeps the real training path: real weights, real log-probabilities,
real `loss.backward()`, one real `optimizer.step()`, a real `save_pretrained` and
a real reload. This module only decides whether that path is *allowed* to run and
whether it *did what it claimed*.

Two rules this module exists to enforce
---------------------------------------
**Measure, do not assert.** An earlier draft of the smoke reported
``optimizer_step_completed: True`` as a literal. A boolean that is written by the
same function that would be wrong is not evidence. Every claim here is a value
that was computed from the tensors or from a counter that was incremented, and
the report carries the computed value, not the intent.

**Refuse before the expensive step.** Loading a 3B model takes seconds and
allocates gigabytes. Every refusal that can be decided from the arguments and the
input rows happens first, so a mistyped invocation costs nothing and leaves no
GPU state behind. `assert_phase7_model_is_permitted` is called before any loader
is constructed, and the tests use a loader spy to prove it (not just to claim it).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = [
    "PERMITTED_MODEL_NAME",
    "PERMITTED_MODEL_REVISION",
    "PERMITTED_REWARDS",
    "REQUIRED_ROLLOUT_COUNT",
    "EXPECTED_OPTIMIZER_STEPS",
    "PERMITTED_ITERATIONS",
    "KL_STATUS_NOT_APPLICABLE",
    "Phase7Violation",
    "Phase7ExecutionPlan",
    "TrainingTensor",
    "TensorDigest",
    "verify_model_identity",
    "assert_phase7_model_is_permitted",
    "load_permitted_model",
    "load_permitted_tokenizer",
    "validate_rollout_group",
    "validate_execution_plan",
    "assert_lora_only_trainable",
    "base_tensor_manifest",
    "tensor_digest",
    "compare_base_manifests",
    "count_changed_tensors",
    "OptimizerStepCounter",
]

#: The only model Phase 7 may load. ADR-0008 forbids the 1.5B; standing policy
#: forbids the 7B outright. There is no override argument, by design.
PERMITTED_MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"

#: The immutable revision. Phase 6M ran this exact snapshot; a different one is a
#: different experiment and the manifest would be lying about which weights moved.
PERMITTED_MODEL_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"

#: The Phase 6M group in canonical `rollout_index` order. Not sorted, not
#: deduplicated, not relaxed to a multiset: the order *is* the identity of the
#: rows the advantage vector belongs to. r00 solved the task; the other three did
#: not, which is the whole reason this group carries any learning signal.
PERMITTED_REWARDS: tuple[float, ...] = (0.0, -1.0, -1.0, -1.0)

#: A GRPO group of four. Four is not a hyperparameter here; it is the Phase 6M
#: group cardinality, and a runner that grew to five would be a different run.
REQUIRED_ROLLOUT_COUNT = 4

#: Exactly one update. Named rather than inlined so the tests can assert the
#: refusal message quotes the same number the enforcement uses.
EXPECTED_OPTIMIZER_STEPS = 1
PERMITTED_ITERATIONS = 1

#: Report text for the KL column. The current implementation has no reference
#: policy, so there is no KL to report. Writing 0.0 would claim a measured
#: quantity that was never measured.
KL_STATUS_NOT_APPLICABLE = "NOT_APPLICABLE_CURRENT_IMPLEMENTATION"


class Phase7Violation(RuntimeError):
    """A Phase 7 precondition failed. Raised before GPU work wherever possible."""


# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------


def verify_model_identity(model_name: Any, revision: Any) -> tuple[str, str]:
    """Return ``(name, revision)`` only if they are the Phase 7 pin.

    Accepts anything and refuses anything that is not exactly the pin, so a
    caller cannot pass a truthy-looking surrogate. ``7B`` is not special-cased
    into its own branch: the permitted set has one member, so the 7B is rejected
    by the same comparison every other wrong model is rejected by. A separate
    "is it 7B" check would be a second place to get wrong.
    """
    if not isinstance(model_name, str) or model_name != PERMITTED_MODEL_NAME:
        raise Phase7Violation(
            "Phase 7 permits only "
            f"{PERMITTED_MODEL_NAME!r}; requested {model_name!r}. "
            "The 1.5B is excluded by ADR-0008 and the 7B is permanently prohibited."
        )

    if not isinstance(revision, str) or revision != PERMITTED_MODEL_REVISION:
        raise Phase7Violation(
            "Phase 7 requires the immutable revision "
            f"{PERMITTED_MODEL_REVISION!r}; requested {revision!r}. "
            "A cache-state resolution is not a pinned revision."
        )

    return model_name, revision


def assert_phase7_model_is_permitted(
    model_name: Any,
    revision: Any,
    *,
    loader: Any = None,
    tokenizer_loader: Any = None,
    cuda_initialised: Any = None,
) -> tuple[str, str]:
    """Refuse a non-permitted model *and prove nothing expensive was touched*.

    The callbacks exist so a test can assert that a refused run never reached a
    loader and never initialised CUDA. In production they are the real
    `torch.cuda.is_initialized`, which is how the script states the ordering
    requirement as an executable fact instead of a comment.
    """
    name, revision = verify_model_identity(model_name, revision)

    if loader is not None and getattr(loader, "call_count", 0):
        raise Phase7Violation("Model loader was invoked before the Phase 7 model guard.")

    if tokenizer_loader is not None and getattr(tokenizer_loader, "call_count", 0):
        raise Phase7Violation("Tokenizer loader was invoked before the Phase 7 model guard.")

    if cuda_initialised is not None and bool(cuda_initialised()):
        raise Phase7Violation("CUDA was initialised before the Phase 7 model guard.")

    return name, revision


def load_permitted_model(
    model_loader: Any,
    *,
    model_name: Any,
    revision: Any,
    **loader_kwargs: Any,
) -> Any:
    """Call ``model_loader`` with the pinned revision, or refuse.

    The revision is passed explicitly *and* the refusal happens first, so the
    guard holds even if a future loader stops honouring the argument. Tested with
    a spy loader: on refusal, ``call_count`` stays zero.
    """
    name, pinned_revision = verify_model_identity(model_name, revision)
    return model_loader(name, revision=pinned_revision, **loader_kwargs)


def load_permitted_tokenizer(
    tokenizer_loader: Any,
    *,
    model_name: Any,
    revision: Any,
    **loader_kwargs: Any,
) -> Any:
    """Tokenizer counterpart of :func:`load_permitted_model`.

    The tokenizer is pinned too. A tokenizer from another revision would not
    change the weights, and would silently change every token ID in the run.
    """
    name, pinned_revision = verify_model_identity(model_name, revision)
    return tokenizer_loader(name, revision=pinned_revision, **loader_kwargs)


# ---------------------------------------------------------------------------
# Input group
# ---------------------------------------------------------------------------


def _reward_of(row: Mapping[str, Any]) -> float:
    reward = row.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise Phase7Violation(f"rollout reward must be a number, got {reward!r}")
    return float(reward)


# The recorded trace nests the fields the guard needs: `agent.rollout_index`,
# `result.reward`, `agent.sampling_seed`, `task.id`. Earlier code assumed the
# group fields were top-level, which fails against the real Phase 6M file. These
# accessors accept the nested shape and fall back to the flat shape, so the guard
# and the manifest describe the same rows the same way.


def rollout_index_of(row: Mapping[str, Any]) -> Any:
    if "rollout_index" in row:
        return row["rollout_index"]
    agent = row.get("agent")
    if isinstance(agent, Mapping) and "rollout_index" in agent:
        return agent["rollout_index"]
    raise Phase7Violation("rollout row has no rollout_index (checked top level and agent)")


def reward_of(row: Mapping[str, Any]) -> float:
    if "reward" in row:
        return _reward_of(row)
    result = row.get("result")
    if isinstance(result, Mapping) and "reward" in result:
        return _reward_of(result)
    raise Phase7Violation("rollout row has no reward (checked top level and result)")


def seed_of(row: Mapping[str, Any]) -> Any:
    if "sampling_seed" in row:
        return row["sampling_seed"]
    agent = row.get("agent")
    if isinstance(agent, Mapping) and "sampling_seed" in agent:
        return agent["sampling_seed"]
    return None


def task_id_of(row: Mapping[str, Any]) -> str:
    if isinstance(row.get("task_id"), str):
        return row["task_id"]
    task = row.get("task")
    if isinstance(task, Mapping) and isinstance(task.get("id"), str):
        return task["id"]
    raise Phase7Violation("rollout row has no task id (checked task_id and task.id)")


def normalise_for_guard(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project recorded trace rows onto the flat shape `validate_rollout_group` expects.

    This is a projection, not a rewrite: it copies the identity fields the guard
    reads and records the same eligibility rule the trainer applies (a row that
    carries an infrastructure error is not a training candidate). The original
    rows are never mutated or reordered.
    """
    return [
        {
            "task_id": task_id_of(row),
            "rollout_index": rollout_index_of(row),
            "reward": reward_of(row),
            "eligible_for_training": row.get("error") is None,
        }
        for row in rows
    ]


def validate_rollout_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_task_id: str | None = None,
) -> list[Mapping[str, Any]]:
    """Refuse anything but the four Phase 6M rows, in the recorded order.

    Rejects on: wrong cardinality (both directions), more than one task, a
    non-eligible row, a duplicate `rollout_index`, a missing `rollout_index`, or
    a reward vector that differs from the canonical sequence *in order*.

    The order check is deliberately not a multiset comparison. Swapping ``0.0``
    to position 3 leaves the sorted rewards identical and the advantage vector
    attached to the wrong rollout, which is the kind of quiet corruption that
    still produces a plausible loss curve.
    """
    if len(rows) != REQUIRED_ROLLOUT_COUNT:
        direction = "fewer" if len(rows) < REQUIRED_ROLLOUT_COUNT else "more"
        raise Phase7Violation(
            f"Phase 7 requires exactly {REQUIRED_ROLLOUT_COUNT} rollout rows; got {len(rows)} ({direction}). "
            "Dropping or adding a group member changes the advantages, so neither is allowed."
        )

    task_ids: set[Any] = set()
    indices: list[Any] = []
    rewards: list[float] = []

    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise Phase7Violation(f"rollout row {position} is not a mapping")

        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise Phase7Violation(f"rollout row {position} has no task_id")
        task_ids.add(task_id)

        if "rollout_index" not in row:
            raise Phase7Violation(f"rollout row {position} has no rollout_index")
        indices.append(row["rollout_index"])

        if row.get("eligible_for_training") is False:
            raise Phase7Violation(
                f"rollout_index={row['rollout_index']!r} is not eligible for training; "
                "an ineligible row must not reach the optimizer."
            )

        rewards.append(_reward_of(row))

    if len(task_ids) != 1:
        raise Phase7Violation(f"Phase 7 requires exactly one task group; found {sorted(map(str, task_ids))}")

    if expected_task_id is not None and task_ids != {expected_task_id}:
        raise Phase7Violation(f"expected task {expected_task_id!r}; found {sorted(map(str, task_ids))}")

    if len(set(indices)) != len(indices):
        raise Phase7Violation(f"rollout_index values must be unique; found {indices}")

    if len(set(rewards)) < 2:
        raise Phase7Violation(
            f"rollout rewards are uniform, so the group is degenerate and has no relative signal: {rewards}"
        )

    if list(rewards) != list(PERMITTED_REWARDS):
        raise Phase7Violation(
            f"reward sequence must be {list(PERMITTED_REWARDS)} in rollout_index order; got {rewards}"
        )

    return list(rows)


# ---------------------------------------------------------------------------
# Execution plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Phase7ExecutionPlan:
    """Every setting that decides whether this is a Phase 7 run or something else.

    Defaults are the permitted values, so the dataclass itself is a correct
    plan; the validator then becomes the single place that says what is *not*
    allowed. Anything a caller might plausibly toggle is represented, because a
    knob that exists but is not checked is an undocumented feature.
    """

    iterations: int = PERMITTED_ITERATIONS
    planned_optimizer_steps: int = EXPECTED_OPTIMIZER_STEPS
    generate_rollouts: bool = False
    generate_evaluation: bool = False
    heldout_evaluation: bool = False
    adapter_checkpoint: str | None = None
    resume_adapter: bool = False
    model_name: str = PERMITTED_MODEL_NAME
    model_revision: str = PERMITTED_MODEL_REVISION


def validate_execution_plan(plan: Phase7ExecutionPlan) -> None:
    """Refuse a plan that would do more than one update, or generate anything new."""
    if plan.iterations != PERMITTED_ITERATIONS:
        raise Phase7Violation(f"Phase 7 permits exactly {PERMITTED_ITERATIONS} iteration; plan has {plan.iterations}.")

    if plan.planned_optimizer_steps != EXPECTED_OPTIMIZER_STEPS:
        raise Phase7Violation(
            f"Phase 7 permits exactly {EXPECTED_OPTIMIZER_STEPS} optimizer step; "
            f"plan has {plan.planned_optimizer_steps}."
        )

    if plan.generate_rollouts:
        raise Phase7Violation(
            "Phase 7 consumes recorded Phase 6M rollouts and must not generate new ones; "
            "a fresh rollout would be new evidence outside the authorised run."
        )

    if plan.generate_evaluation:
        raise Phase7Violation("Phase 7 must not run evaluation generation.")

    if plan.heldout_evaluation:
        raise Phase7Violation(
            "Held-out evaluation is prohibited during training; a held-out score seen "
            "during Phase 7 would make the task ineligible for the final comparison."
        )

    if plan.adapter_checkpoint is not None:
        raise Phase7Violation(
            "Phase 7 starts from a fresh adapter; an existing adapter checkpoint is prohibited "
            f"(got {plan.adapter_checkpoint!r})."
        )

    if plan.resume_adapter:
        raise Phase7Violation("Phase 7 must not resume or continue existing adapter training.")

    verify_model_identity(plan.model_name, plan.model_revision)


# ---------------------------------------------------------------------------
# LoRA-only trainability
# ---------------------------------------------------------------------------


class TrainingTensor(Protocol):
    """The slice of a tensor the integrity checks need.

    Declared structurally so the checks run against synthetic CPU objects in the
    tests and real ``torch.nn.Parameter`` in the script without either importing
    the other's world.
    """

    @property
    def shape(self) -> Any: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def requires_grad(self) -> bool: ...

    def numel(self) -> int: ...

    def detach(self) -> Any: ...

    def to(self, *args: Any, **kwargs: Any) -> Any: ...

    def contiguous(self) -> Any: ...

    def cpu(self) -> Any: ...

    def numpy(self) -> Any: ...

    def type(self, dtype: Any) -> Any: ...


#: Every LoRA-injected module name PEFT uses with this `target_modules` list.
#: `modules_to_save` is deliberately absent: Phase 7 has no classifier head, and
#: allowing the prefix would let a full-precision copy of a base tensor hide here.
_LORA_MARKERS: tuple[str, ...] = ("lora_A.", "lora_B.", "lora_embedding_A.", "lora_embedding_B.")


def _is_lora_parameter(name: str) -> bool:
    return any(marker in name for marker in _LORA_MARKERS)


def assert_lora_only_trainable(
    named_parameters: Iterable[tuple[str, TrainingTensor]],
) -> tuple[int, int]:
    """Assert the trainable set is exactly the adapter, and return both counts.

    Returns ``(trainable_count, frozen_count)`` measured, not asserted. An empty
    trainable set is a violation too: a run with nothing to update would still
    cheerfully report one successful step.
    """
    trainable: list[str] = []
    frozen = 0

    for name, parameter in named_parameters:
        if bool(parameter.requires_grad):
            trainable.append(name)
        else:
            frozen += 1

    if not trainable:
        raise Phase7Violation("No trainable parameters; a LoRA adapter should have some.")

    unexpected = [name for name in trainable if not _is_lora_parameter(name)]

    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise Phase7Violation(
            f"{len(unexpected)} non-LoRA parameter(s) are trainable: {preview}. "
            "Phase 7 updates the adapter only; a trainable base tensor means the base model "
            "can move."
        )

    return len(trainable), frozen


# ---------------------------------------------------------------------------
# Deterministic base-model manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorDigest:
    """Identity and content digest of one tensor."""

    shape: tuple[int, ...]
    dtype: str
    numel: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "numel": self.numel,
            "sha256": self.sha256,
        }


def _tensor_bytes(tensor: TrainingTensor) -> bytes:
    """Canonical CPU bytes for a tensor: float64, little-endian, contiguous.

    Upcasting through float64 before serialising means a bf16 weight and a
    float32 copy of the same value hash the same after the same rounding, which
    keeps the manifest comparable across a reload that reasonably changes dtype.
    It does *not* make two genuinely different tensors collide: bf16 has 8 bits
    of mantissa and float64 has 52, so every bf16 value is exactly representable
    and the upcast is lossless.
    """
    import numpy as np

    array = tensor.detach().to("cpu").float().numpy().astype(np.float64, copy=False)
    return np.ascontiguousarray(array, dtype="<f8").tobytes()


def tensor_digest(tensor: TrainingTensor) -> str:
    """Content digest of one tensor, as a bare hex string.

    Used for the LoRA delta check, where the name is already known and only the
    content matters.
    """
    return hashlib.sha256(_tensor_bytes(tensor)).hexdigest()


def base_tensor_manifest(
    named_parameters: Iterable[tuple[str, TrainingTensor]],
) -> dict[str, TensorDigest]:
    """Digest every non-LoRA tensor. The base model must be byte-identical after the step.

    Copies to CPU one tensor at a time and never retains the tensor, so peak host
    memory is one tensor rather than a duplicate of the model. That is why this
    is a manifest of digests and not a saved copy of the weights.
    """
    manifest: dict[str, TensorDigest] = {}

    for name, parameter in named_parameters:
        if _is_lora_parameter(name):
            continue

        manifest[name] = TensorDigest(
            shape=tuple(int(dimension) for dimension in parameter.shape),
            dtype=str(parameter.dtype),
            numel=int(parameter.numel()),
            sha256=hashlib.sha256(_tensor_bytes(parameter)).hexdigest(),
        )

    if not manifest:
        raise Phase7Violation("Base-model manifest is empty; nothing would be verified frozen.")

    return manifest


def compare_base_manifests(
    before: Mapping[str, TensorDigest],
    after: Mapping[str, TensorDigest],
) -> dict[str, list[str]]:
    """Name every base tensor that was added, removed, reshaped, retyped or changed content.

    Returns the four buckets separately rather than one boolean, because
    "a tensor changed" and "a tensor appeared" have different causes, and the
    report is supposed to be diagnosable without a rerun.
    """
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))

    metadata_changed: list[str] = []
    content_changed: list[str] = []

    for name in sorted(set(before) & set(after)):
        old = before[name]
        new = after[name]

        if (old.shape, old.dtype, old.numel) != (new.shape, new.dtype, new.numel):
            metadata_changed.append(name)
        elif old.sha256 != new.sha256:
            content_changed.append(name)

    return {
        "added": added,
        "removed": removed,
        "metadata_changed": metadata_changed,
        "content_changed": content_changed,
    }


def count_changed_tensors(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> list[str]:
    """Names of tensors that differ in content, for the LoRA delta check.

    Both mappings are ``name -> hashable content digest``, so this is a set
    comparison plus a digest comparison and never a tensor comparison. The
    LoRA tensors are small, but this keeps the two integrity checks symmetric:
    both measure digests, neither trusts a flag.
    """
    missing = sorted(set(before) - set(after))
    if missing:
        raise Phase7Violation(f"{len(missing)} tensor(s) disappeared during the step: {missing[:5]}")

    return sorted(name for name in after if name in before and before[name] != after[name])


# ---------------------------------------------------------------------------
# Optimizer step counter
# ---------------------------------------------------------------------------


@dataclass
class OptimizerStepCounter:
    """A counter that increments only after a step returns normally.

    The point is the ordering: ``optimizer.step()`` must complete before the
    count moves. If the step raises, the counter stays where it was, and the
    assertion after the step records a failure rather than a number that is one
    too high. A module-level boolean cannot express that, which is why the old
    ``optimizer_step_completed: True`` was replaced.
    """

    limit: int = EXPECTED_OPTIMIZER_STEPS
    count: int = 0
    completed_steps: list[int] = field(default_factory=list)

    def assert_zero(self) -> None:
        if self.count != 0:
            raise Phase7Violation(f"optimizer step counter must be 0 before the step; it is {self.count}.")

    def record_completed_step(self) -> int:
        if self.count + 1 > self.limit:
            raise Phase7Violation(f"Phase 7 permits exactly {self.limit} optimizer step; a further step was attempted.")
        self.count += 1
        self.completed_steps.append(self.count)
        return self.count

    def assert_exactly_one(self) -> None:
        if self.count != self.limit:
            raise Phase7Violation(f"optimizer step counter must be {self.limit} after the step; it is {self.count}.")
