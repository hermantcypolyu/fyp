"""Part E: CPU-only preflight tests for the Phase 7 fail-closed guards.

These tests load no model weights, touch no CUDA, contact no network and read no
Hugging Face cache. They are not the Phase 7 result; they are the proof that the
guards standing between a mistyped invocation and a GPU run actually fire.

Two conventions worth knowing before reading them:

**Loader spies, not "no loader is called".** The requirement is that a refused
run never reaches a loader or initialises CUDA. Asserting that inside test code
proving the guard refused would be circular, so the guard takes the loaders and
the CUDA predicate as arguments, and the tests pass spies and assert on
``call_count`` and ``cuda_initialised`` directly.

**Synthetic tensors, real digests.** The base-manifest tests use
``torch.nn.Parameter`` on CPU, because `torch` is present in the interpreter that
runs these tests. Where a test only needs shape/dtype/numel it uses
:class:`FakeTensor`, which keeps the guard arithmetic independent of torch.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is required for the tensor-level Phase 7 guards")

from training.phase7_guards import (  # noqa: E402 - importorskip must run first
    EXPECTED_OPTIMIZER_STEPS,
    KL_STATUS_NOT_APPLICABLE,
    PERMITTED_MODEL_NAME,
    PERMITTED_MODEL_REVISION,
    PERMITTED_REWARDS,
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
    reward_of,
    rollout_index_of,
    seed_of,
    task_id_of,
    validate_execution_plan,
    validate_rollout_group,
    verify_model_identity,
)

TASK_ID = "bugfix_offbyone_pagination"
OTHER_TASK_ID = "bugfix_filesize_unit_rollover"


class LoaderSpy:
    """A loader that records whether anybody called it, and with what."""

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.calls.append((args, kwargs))
        return object()


class FakeTensor:
    """Shape/dtype/requires_grad only. Used where content is irrelevant."""

    def __init__(self, numel: int, *, requires_grad: bool, dtype: str = "torch.float32") -> None:
        self._numel = numel
        self.requires_grad = requires_grad
        self.dtype = dtype
        self.shape = (numel,)

    def numel(self) -> int:
        return self._numel


def make_rollouts(rewards=PERMITTED_REWARDS, *, task_id: str = TASK_ID) -> list[dict]:
    """The canonical four rows, in `rollout_index` order."""
    return [
        {
            "task_id": task_id,
            "rollout_index": index,
            "reward": float(reward),
            "eligible_for_training": True,
        }
        for index, reward in enumerate(rewards)
    ]


def make_adapter_state(rank: int = 2) -> dict[str, torch.nn.Parameter]:
    """A miniature LoRA-wrapped module: base weights frozen, adapters trainable."""
    return {
        "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.nn.Parameter(
            torch.zeros(4, 4), requires_grad=False
        ),
        "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.nn.Parameter(
            torch.linspace(-0.5, 0.5, 4 * rank).reshape(rank, 4)
        ),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.nn.Parameter(
            torch.linspace(0.5, -0.5, rank * 4).reshape(4, rank)
        ),
    }


# ---------------------------------------------------------------------------
# 1-4: model identity, revision and the pre-loader rejection order
# ---------------------------------------------------------------------------


def test_exact_3b_model_and_revision_accepted():
    name, revision = verify_model_identity(PERMITTED_MODEL_NAME, PERMITTED_MODEL_REVISION)

    assert name == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert revision == "488639f1ff808d1d3d0ba301aef8c11461451ec5"


@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-32B-Instruct",
        "Qwen/Qwen2.5-Coder-3B",
        "NotAModel",
    ],
)
def test_wrong_model_rejected(model_name):
    with pytest.raises(Phase7Violation):
        verify_model_identity(model_name, PERMITTED_MODEL_REVISION)


def test_7b_model_rejected_before_loader_invocation():
    """The 7B must be refused without a loader call and without touching CUDA.

    This is the test the prohibition actually needs. A comment saying "no 7B"
    cannot fail; this assertion can.
    """
    model_loader = LoaderSpy()
    tokenizer_loader = LoaderSpy()
    cuda = {"initialised": False}

    with pytest.raises(Phase7Violation):
        assert_phase7_model_is_permitted(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            PERMITTED_MODEL_REVISION,
            loader=model_loader,
            tokenizer_loader=tokenizer_loader,
            cuda_initialised=lambda: cuda["initialised"],
        )

    assert model_loader.call_count == 0
    assert tokenizer_loader.call_count == 0
    assert cuda["initialised"] is False


def test_7b_rejection_survives_load_permitted_model_helper():
    model_loader = LoaderSpy()

    with pytest.raises(Phase7Violation):
        load_permitted_model(
            model_loader, model_name="Qwen/Qwen2.5-Coder-7B-Instruct", revision=PERMITTED_MODEL_REVISION
        )

    assert model_loader.call_count == 0


def test_permitted_model_loader_receives_explicit_revision():
    model_loader = LoaderSpy()

    load_permitted_model(
        model_loader,
        model_name=PERMITTED_MODEL_NAME,
        revision=PERMITTED_MODEL_REVISION,
        local_files_only=True,
    )

    assert model_loader.call_count == 1
    _, kwargs = model_loader.calls[0]
    assert kwargs["revision"] == PERMITTED_MODEL_REVISION
    assert kwargs["local_files_only"] is True


@pytest.mark.parametrize(
    "revision",
    [
        "2e1fd397ee46e1388853d2af2c993145b0f1098a",
        "main",
        "",
        None,
        "488639f1ff808d1d3d0ba301aef8c11461451ec5".upper(),
    ],
)
def test_wrong_revision_rejected(revision):
    with pytest.raises(Phase7Violation):
        verify_model_identity(PERMITTED_MODEL_NAME, revision)


def test_tokenizer_loader_is_pinned_too():
    tokenizer_loader = LoaderSpy()

    load_permitted_tokenizer(
        tokenizer_loader,
        model_name=PERMITTED_MODEL_NAME,
        revision=PERMITTED_MODEL_REVISION,
        local_files_only=True,
    )

    _, kwargs = tokenizer_loader.calls[0]
    assert kwargs["revision"] == PERMITTED_MODEL_REVISION

    with pytest.raises(Phase7Violation):
        load_permitted_tokenizer(
            tokenizer_loader,
            model_name=PERMITTED_MODEL_NAME,
            revision="main",
        )


# ---------------------------------------------------------------------------
# 5-6: iterations and optimizer steps
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("iterations", [0, 2, 5])
def test_iterations_other_than_one_rejected(iterations):
    with pytest.raises(Phase7Violation):
        validate_execution_plan(Phase7ExecutionPlan(iterations=iterations))


@pytest.mark.parametrize("steps", [0, 2, 4])
def test_planned_optimizer_steps_other_than_one_rejected(steps):
    with pytest.raises(Phase7Violation):
        validate_execution_plan(Phase7ExecutionPlan(planned_optimizer_steps=steps))


def test_permitted_plan_is_accepted_and_reports_kl_not_applicable():
    validate_execution_plan(Phase7ExecutionPlan())

    assert EXPECTED_OPTIMIZER_STEPS == 1
    # The implementation has no reference policy, so there is no KL to report.
    assert KL_STATUS_NOT_APPLICABLE == "NOT_APPLICABLE_CURRENT_IMPLEMENTATION"


# ---------------------------------------------------------------------------
# 7: multiple task groups
# ---------------------------------------------------------------------------


def test_multiple_task_groups_rejected():
    rows = make_rollouts()
    rows[3]["task_id"] = OTHER_TASK_ID

    with pytest.raises(Phase7Violation, match="one task group"):
        validate_rollout_group(rows)


def test_unexpected_task_id_rejected():
    with pytest.raises(Phase7Violation, match="expected task"):
        validate_rollout_group(make_rollouts(), expected_task_id=OTHER_TASK_ID)


# ---------------------------------------------------------------------------
# 8-9: cardinality
# ---------------------------------------------------------------------------


def test_fewer_than_four_rows_rejected():
    rows = make_rollouts()[:3]

    with pytest.raises(Phase7Violation, match="exactly 4"):
        validate_rollout_group(rows)


def test_more_than_four_rows_rejected():
    rows = make_rollouts() + [{"task_id": TASK_ID, "rollout_index": 4, "reward": -1.0, "eligible_for_training": True}]

    with pytest.raises(Phase7Violation, match="exactly 4"):
        validate_rollout_group(rows)


def test_canonical_group_accepted_unchanged():
    rows = make_rollouts()
    accepted = validate_rollout_group(rows, expected_task_id=TASK_ID)

    assert accepted == rows
    assert [row["rollout_index"] for row in accepted] == [0, 1, 2, 3]
    assert [row["reward"] for row in accepted] == [0.0, -1.0, -1.0, -1.0]


def test_permuted_rewards_rejected_even_though_multiset_matches():
    """Order is identity: the same four rewards attached to the wrong rollouts.

    `sorted(before) == sorted(after)`, so a multiset check would pass this. It is
    still the wrong group, because r00 is the rollout that solved the task.
    """
    rows = make_rollouts(rewards=(-1.0, 0.0, -1.0, -1.0))

    assert sorted(row["reward"] for row in rows) == sorted(PERMITTED_REWARDS)

    with pytest.raises(Phase7Violation, match="rollout_index order"):
        validate_rollout_group(rows)


def test_duplicate_rollout_index_rejected():
    rows = make_rollouts()
    rows[2]["rollout_index"] = 1

    with pytest.raises(Phase7Violation, match="unique"):
        validate_rollout_group(rows)


def test_ineligible_row_rejected():
    rows = make_rollouts()
    rows[1]["eligible_for_training"] = False

    with pytest.raises(Phase7Violation, match="not eligible"):
        validate_rollout_group(rows)


# ---------------------------------------------------------------------------
# Recorded trace shape
# ---------------------------------------------------------------------------
#
# The real Phase 6M rows nest the fields the guard reads: `agent.rollout_index`,
# `result.reward`, `agent.sampling_seed`, `task.id`. A version of this guard that
# only understood flat rows passed its own tests and then refused the real file
# at run time, so the nested shape gets tests of its own.


def make_trace_row(index: int, reward: float) -> dict:
    return {
        "rollout_id": f"phase6m__r{index:02d}",
        "task": {"id": TASK_ID, "split": "train"},
        "agent": {"rollout_index": index, "sampling_seed": 1000 + index},
        "result": {"reward": reward},
        "error": None,
    }


def make_trace_rows(rewards=PERMITTED_REWARDS) -> list[dict]:
    return [make_trace_row(index, reward) for index, reward in enumerate(rewards)]


def test_accessors_read_nested_trace_fields():
    row = make_trace_row(2, -1.0)

    assert rollout_index_of(row) == 2
    assert reward_of(row) == -1.0
    assert seed_of(row) == 1002
    assert task_id_of(row) == TASK_ID


def test_normalise_projects_nested_rows_onto_the_guard_shape():
    flat = normalise_for_guard(make_trace_rows())

    assert [row["rollout_index"] for row in flat] == [0, 1, 2, 3]
    assert [row["reward"] for row in flat] == [0.0, -1.0, -1.0, -1.0]
    assert all(row["eligible_for_training"] is True for row in flat)


def test_nested_canonical_group_passes_the_guard():
    """The exact failure that stopped the first GPU attempt: real rows, real shape."""
    validated = validate_rollout_group(normalise_for_guard(make_trace_rows()))

    assert len(validated) == 4


def test_nested_permuted_rewards_still_rejected():
    rows = make_trace_rows(rewards=(-1.0, 0.0, -1.0, -1.0))

    with pytest.raises(Phase7Violation, match="rollout_index order"):
        validate_rollout_group(normalise_for_guard(rows))


def test_nested_row_with_error_is_ineligible():
    rows = make_trace_rows()
    rows[1]["error"] = "vllm timeout"

    with pytest.raises(Phase7Violation, match="not eligible"):
        validate_rollout_group(normalise_for_guard(rows))


def test_missing_nested_fields_raise_rather_than_defaulting():
    """A row whose identity cannot be read must not silently become index-by-position."""
    row = make_trace_row(0, 0.0)
    del row["agent"]

    with pytest.raises(Phase7Violation, match="no rollout_index"):
        rollout_index_of(row)


def test_missing_task_id_raises_rather_than_defaulting():
    row = make_trace_row(0, 0.0)
    del row["task"]

    with pytest.raises(Phase7Violation, match="no task id"):
        task_id_of(row)


def test_missing_reward_raises_rather_than_defaulting():
    row = make_trace_row(0, 0.0)
    del row["result"]

    with pytest.raises(Phase7Violation, match="no reward"):
        reward_of(row)


# ---------------------------------------------------------------------------
# 10-13: prohibited execution paths
# ---------------------------------------------------------------------------


def test_rollout_generation_rejected():
    with pytest.raises(Phase7Violation, match="must not generate"):
        validate_execution_plan(Phase7ExecutionPlan(generate_rollouts=True))


def test_evaluation_generation_rejected():
    with pytest.raises(Phase7Violation, match="evaluation generation"):
        validate_execution_plan(Phase7ExecutionPlan(generate_evaluation=True))


def test_heldout_evaluation_rejected():
    with pytest.raises(Phase7Violation, match="Held-out"):
        validate_execution_plan(Phase7ExecutionPlan(heldout_evaluation=True))


def test_existing_adapter_input_rejected(tmp_path):
    with pytest.raises(Phase7Violation, match="existing adapter"):
        validate_execution_plan(Phase7ExecutionPlan(adapter_checkpoint=str(tmp_path)))


def test_adapter_resume_rejected():
    with pytest.raises(Phase7Violation, match="resume"):
        validate_execution_plan(Phase7ExecutionPlan(resume_adapter=True))


def test_plan_with_wrong_model_rejected():
    with pytest.raises(Phase7Violation):
        validate_execution_plan(Phase7ExecutionPlan(model_name="Qwen/Qwen2.5-Coder-7B-Instruct"))


# ---------------------------------------------------------------------------
# 14: LoRA-only trainability
# ---------------------------------------------------------------------------


def test_non_lora_trainable_base_parameter_rejected():
    state = make_adapter_state()
    # A base tensor that somebody unfroze. Nothing else about the run is wrong.
    state["base_model.model.layers.0.self_attn.q_proj.base_layer.weight"].requires_grad_(True)

    with pytest.raises(Phase7Violation, match="non-LoRA"):
        assert_lora_only_trainable(state.items())


def test_lora_only_state_accepted_with_measured_counts():
    trainable_count, frozen_count = assert_lora_only_trainable(make_adapter_state().items())

    assert trainable_count == 2
    assert frozen_count == 1


def test_empty_trainable_set_rejected():
    frozen_only = {
        "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.nn.Parameter(
            torch.zeros(4, 4), requires_grad=False
        )
    }

    with pytest.raises(Phase7Violation, match="No trainable parameters"):
        assert_lora_only_trainable(frozen_only.items())


def test_modules_to_save_prefix_is_not_treated_as_lora():
    """A `modules_to_save` copy of a base tensor must not masquerade as an adapter."""
    state = {
        **make_adapter_state(),
        "base_model.model.layers.0.self_attn.q_proj.modules_to_save.default.weight": torch.nn.Parameter(
            torch.zeros(4, 4)
        ),
    }

    with pytest.raises(Phase7Violation, match="non-LoRA"):
        assert_lora_only_trainable(state.items())


# ---------------------------------------------------------------------------
# 15-17: base-model manifest
# ---------------------------------------------------------------------------


def test_unchanged_base_tensor_manifest_accepted():
    state = make_adapter_state()

    before = base_tensor_manifest(state.items())
    after = base_tensor_manifest(state.items())
    changes = compare_base_manifests(before, after)

    assert changes == {"added": [], "removed": [], "metadata_changed": [], "content_changed": []}
    assert len(before) == 1  # only the frozen base tensor; the two adapters are excluded


def test_base_tensor_mutation_detected():
    state = make_adapter_state()
    before = base_tensor_manifest(state.items())

    with torch.no_grad():
        state["base_model.model.layers.0.self_attn.q_proj.base_layer.weight"].add_(1e-3)

    changes = compare_base_manifests(before, base_tensor_manifest(state.items()))

    assert changes["content_changed"] == ["base_model.model.layers.0.self_attn.q_proj.base_layer.weight"]
    assert changes["added"] == []
    assert changes["removed"] == []


def test_base_tensor_metadata_change_detected_separately_from_content():
    state = make_adapter_state()
    before = base_tensor_manifest(state.items())

    state["base_model.model.layers.0.self_attn.q_proj.base_layer.weight"].data = torch.zeros(2, 8)

    changes = compare_base_manifests(before, base_tensor_manifest(state.items()))

    assert changes["metadata_changed"] == ["base_model.model.layers.0.self_attn.q_proj.base_layer.weight"]
    assert changes["content_changed"] == []


def test_missing_base_tensor_reported_by_name():
    """A base tensor that vanishes is reported as removed, not silently ignored."""
    state = make_adapter_state()
    before = base_tensor_manifest(state.items())

    after = base_tensor_manifest(make_adapter_state().items())
    del after["base_model.model.layers.0.self_attn.q_proj.base_layer.weight"]

    changes = compare_base_manifests(before, after)

    assert changes["removed"] == ["base_model.model.layers.0.self_attn.q_proj.base_layer.weight"]
    assert changes["added"] == []
    assert changes["content_changed"] == []
    assert changes["metadata_changed"] == []


def test_empty_base_manifest_rejected():
    only_adapters = {name: parameter for name, parameter in make_adapter_state().items() if "lora_" in name}

    with pytest.raises(Phase7Violation, match="manifest is empty"):
        base_tensor_manifest(only_adapters.items())


def test_digest_changes_when_content_changes_bitwise():
    """The digest must respond to a change far below bf16 resolution."""
    tensor = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))

    digest_before = base_tensor_manifest({"w": tensor}.items())["w"].sha256

    with torch.no_grad():
        tensor.add_(1e-12)

    digest_after = base_tensor_manifest({"w": tensor}.items())["w"].sha256

    assert digest_before != digest_after


# ---------------------------------------------------------------------------
# 18-20: LoRA delta and the step counter
# ---------------------------------------------------------------------------


def test_zero_lora_changes_rejected():
    adapters = {
        name: base_tensor_manifest({"t": parameter}.items())["t"].sha256
        for name, parameter in make_adapter_state().items()
        if "lora_" in name
    }

    assert count_changed_tensors(adapters, dict(adapters)) == []


def test_lora_change_reported_by_name():
    state = make_adapter_state()
    before = {
        name: base_tensor_manifest({"t": parameter}.items())["t"].sha256
        for name, parameter in state.items()
        if "lora_" in name
    }

    with torch.no_grad():
        state["base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"].add_(0.25)

    after = {
        name: base_tensor_manifest({"t": parameter}.items())["t"].sha256
        for name, parameter in state.items()
        if "lora_" in name
    }

    assert count_changed_tensors(before, after) == ["base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"]


def test_missing_tensor_during_step_is_a_violation():
    with pytest.raises(Phase7Violation, match="disappeared"):
        count_changed_tensors({"a": "x"}, {})


def test_second_optimizer_step_rejected():
    counter = OptimizerStepCounter()
    counter.assert_zero()

    counter.record_completed_step()
    counter.assert_exactly_one()

    with pytest.raises(Phase7Violation, match="further step"):
        counter.record_completed_step()


def test_step_counter_stays_zero_when_the_step_raises():
    """The counter must not claim a step that did not finish."""
    counter = OptimizerStepCounter()
    counter.assert_zero()

    def failing_step() -> None:
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError, match="out of memory"):
        failing_step()
        counter.record_completed_step()

    assert counter.count == 0
    counter.assert_zero()


def test_step_counter_asserts_one_after_no_step():
    with pytest.raises(Phase7Violation, match="must be 1"):
        OptimizerStepCounter().assert_exactly_one()
