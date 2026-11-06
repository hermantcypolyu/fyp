"""Deterministic per-rollout sampling seeds.

Why this module exists
----------------------
Phase 6G collected four rollouts and got four identical trajectories. The cause was not
the model: `collect_baseline_rollouts.py` sent `seed=cfg.seed` on every completion call, so
`sampling.seed: 42` was a *per-request* constant and every rollout in the batch was the same
deterministic sample. `run_one_rollout` received `rollout_index` and never read it.

What that does and does not establish matters for the record. Phase 6G cannot establish the
model's reward distribution, because all four attempts reused the same deterministic seed and
produced duplicate trajectories: one observation repeated four times is not a sample of four.
The result proves a sampling-independence defect in the harness. It does not show that the
model would do better - the model may still produce uniform -1.0 rewards once independent
seeds are applied, and only a run with independent seeds can tell the two apart.

That is not a cosmetic bug. A "four-rollout baseline" whose four rollouts are one rollout
measured four times cannot support a variance estimate, cannot give GRPO a within-group
signal, and cannot be distinguished from a model that simply collapses. Worse, the failure
is silent: the rows are well-formed, the summary is well-formed, and only a hash of the
candidate patches reveals that nothing varied.

The fix has to satisfy two properties that pull against each other:

  * **Independent** - different rollouts of the same task must sample differently, or there
    is nothing to average over.
  * **Reproducible** - a rollout must be re-runnable from its own row. A seed drawn from
    `random` or `os.urandom` would satisfy independence and destroy reproducibility, which
    is the whole point of recording a seed.

Deriving the seed from `(base_seed, task_id, rollout_index)` satisfies both: the inputs are
all recorded, so the seed is recomputable, and the inputs differ per rollout, so the samples
differ. `hash()` is not used anywhere here - CPython salts `hash()` of `str` per process
(PYTHONHASHSEED), so a seed built from it would differ between the collector and any later
re-verification in a fresh process. SHA-256 has no such salt.

The output is deliberately narrowed to **unsigned 32-bit**. vLLM's OpenAI-compatible server
passes the seed through to numpy/torch RNGs whose accepted ranges differ by version; a value
above 2**32-1 either raises or is silently truncated depending on the build. Constraining
the range here means the recorded seed is always the seed the server actually used.

Typical use
-----------
    >>> derive_rollout_seed(42, "bugfix_offbyone_pagination", 0)
    184288048
    >>> derive_rollout_seed(42, "bugfix_offbyone_pagination", 0) == derive_rollout_seed(
    ...     42, "bugfix_offbyone_pagination", 0
    ... )
    True
    >>> len({derive_rollout_seed(42, "bugfix_offbyone_pagination", i) for i in range(4)})
    4

Nothing here touches a GPU, a model, a server or the network.
"""

from __future__ import annotations

import hashlib
from typing import Final

#: Version label written into every artifact that records a derived seed. It is part of the
#: derivation contract: changing the digest input, the field order, the encoding or the
#: output width is a *breaking* change and must bump this label, because the same
#: `(base_seed, task_id, rollout_index)` would otherwise map to a different seed. A resume
#: or a re-verification that finds a different label than the run was collected under must
#: treat the batch as unreproducible rather than silently re-deriving.
SAMPLING_SEED_DERIVATION: Final[str] = "sha256_u32_v1"

#: Domain separator. Prepended to the digest input so that a seed derived by *this* scheme
#: can never coincide with a seed derived by some future scheme that hashes a similar tuple,
#: and so the purpose of the digest is legible in the bytes rather than only in this file.
_DOMAIN: Final[bytes] = b"fyp.sampling.rollout_seed.v1\x00"

#: Inclusive upper bound. `2**32 - 1`, the largest value a uint32 can hold.
UINT32_MAX: Final[int] = 2**32 - 1


class SeedDerivationError(ValueError):
    """A seed could not be derived, or a planned seed mapping is not usable."""


def derive_rollout_seed(base_seed: int, task_id: str, rollout_index: int) -> int:
    """The sampling seed for one `(task_id, rollout_index)` under `base_seed`.

    Stable across processes, machines, platforms and Python versions: the input is a
    length-prefixed UTF-8 encoding of the three fields, digested with SHA-256. No Python
    built-in `hash()` participates, so PYTHONHASHSEED cannot change the result.

    The fields are length-prefixed rather than merely joined by a separator because a
    separator is only unambiguous while no field can contain it. Task ids are free-form
    strings, so a task literally named `a:b` plus index 1 would otherwise be
    indistinguishable from a task named `a` plus indices `b:1`.
    """
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise SeedDerivationError(f"base_seed must be an int, not {type(base_seed).__name__}")
    if base_seed < 0:
        raise SeedDerivationError(f"base_seed must be non-negative, got {base_seed}")
    if not isinstance(rollout_index, int) or isinstance(rollout_index, bool):
        raise SeedDerivationError(f"rollout_index must be an int, not {type(rollout_index).__name__}")
    if rollout_index < 0:
        raise SeedDerivationError(f"rollout_index must be non-negative, got {rollout_index}")
    if not isinstance(task_id, str):
        raise SeedDerivationError(f"task_id must be a str, not {type(task_id).__name__}")
    if not task_id:
        raise SeedDerivationError("task_id must not be empty")

    payload = b"".join(
        (
            _DOMAIN,
            _field(base_seed),
            _field(task_id),
            _field(rollout_index),
        )
    )
    digest = hashlib.sha256(payload).digest()
    # First four bytes, big-endian. Taking the *first* bytes rather than the last matches
    # every other "sha256 to u32" convention in the ecosystem, so a reader who reaches for
    # a one-liner elsewhere reproduces these numbers instead of almost reproducing them.
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def derive_turn_seed(rollout_seed: int, turn_index: int) -> int:
    """The sampling seed for one turn of one rollout.

    Only used when a per-turn seed is required. The serving API `seed` field is a
    *request-level* seed, so sending one seed for every turn is the simpler policy and is
    the default here; this function exists for the case where a caller wants every turn to
    vary independently while the *rollout* as a whole stays reproducible from
    `(base_seed, task_id, rollout_index)`.

    The derivation is the same construction with the rollout seed in place of the base
    seed and the turn index in place of the rollout index, so there is one algorithm to
    verify rather than two.
    """
    if not isinstance(turn_index, int) or isinstance(turn_index, bool):
        raise SeedDerivationError(f"turn_index must be an int, not {type(turn_index).__name__}")
    if turn_index < 0:
        raise SeedDerivationError(f"turn_index must be non-negative, got {turn_index}")
    return derive_rollout_seed(rollout_seed, "turn", turn_index)


def _field(value: int | str) -> bytes:
    """Length-prefix a field so concatenation is unambiguous.

    The tag distinguishes the two types that could otherwise encode to the same bytes: the
    integer `1` and the string `"1"` both become `b"1"`, and a task id of `"1"` at index 2
    must not collide with some other pair.
    """
    raw = value.encode("utf-8") if isinstance(value, str) else str(value).encode("ascii")
    tag = b"s" if isinstance(value, str) else b"i"
    return tag + len(raw).to_bytes(4, "big") + raw


def plan_seeds(
    base_seed: int,
    task_ids: list[str] | tuple[str, ...],
    rollouts_per_task: int,
) -> list[dict[str, object]]:
    """The full `(task_id, rollout_index) -> sampling_seed` mapping for a planned run.

    Returned sorted by `(task_id, rollout_index)` so the plan is byte-stable: the same
    inputs produce the same list in the same order, which is what lets the manifest record
    a hash of the plan instead of every entry.
    """
    if rollouts_per_task < 1:
        raise SeedDerivationError(f"rollouts_per_task must be >= 1, got {rollouts_per_task}")
    plan: list[dict[str, object]] = []
    for task_id in sorted(task_ids):
        for index in range(rollouts_per_task):
            plan.append(
                {
                    "task_id": task_id,
                    "rollout_index": index,
                    "sampling_seed": derive_rollout_seed(base_seed, task_id, index),
                }
            )
    return plan


def assert_unique_seeds(plan: list[dict[str, object]]) -> None:
    """Reject a plan in which two rollouts would sample from the same seed.

    A collision is astronomically unlikely for a handful of rollouts, which is exactly why
    it must be checked rather than assumed: if it ever happened, the symptom would be the
    Phase 6G symptom - two "independent" rollouts that are silently the same sample - and
    it would be discovered only after burning the GPU time. Checking costs nothing and
    fails before the first model call.
    """
    seen: dict[int, tuple[str, int]] = {}
    for entry in plan:
        seed = int(entry["sampling_seed"])  # type: ignore[arg-type]
        key = (str(entry["task_id"]), int(entry["rollout_index"]))  # type: ignore[arg-type]
        if seed in seen:
            raise SeedDerivationError(
                f"duplicate sampling_seed {seed} planned for {key[0]} rollout {key[1]}; "
                f"it collides with {seen[seed][0]} rollout {seen[seed][1]}. Two rollouts "
                "sharing a seed are the same sample recorded twice, not two samples."
            )
        seen[seed] = key


def plan_digest(plan: list[dict[str, object]]) -> str:
    """A stable hex digest of a seed plan, for recording in a manifest.

    Records *which* mapping was planned without copying every row into the manifest. Phase
    6H writes both: the digest for cheap comparison, and the mapping itself for the selected
    tasks, because a digest alone cannot be inverted by a reader.
    """
    canonical = "|".join(f"{e['task_id']}:{e['rollout_index']}:{e['sampling_seed']}" for e in plan)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
