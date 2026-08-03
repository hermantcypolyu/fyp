"""`reward_v1`, normalised against the measured per-task floor (ADR 0006).

Raw pass fraction made a do-nothing rollout worth the task's floor - 0.46 on average across
this corpus, 0.58 at worst. That is not merely an offset. Doing nothing, editing the wrong
file, and running out of turns all score *exactly* the same number, so a GRPO group made of
those rollouts has zero variance and contributes no gradient at all.

Normalising against the floor puts the plateau at a value that means what it says:

    reward = (passed - floor_tests) / (n_tests - floor_tests)

    passed == floor_tests  ->  0.0   nothing achieved
    passed == n_tests      ->  1.0   solved
    passed <  floor_tests  ->  < 0   actively made it worse

The denominator cannot be zero: `validate_tasks.check_behaviour` rejects any task whose
untouched repo already passes every hidden test (`NOT_FAILABLE`).

**Why the gate constant lives in this module, next to the formula.** Under the raw scale a
gate yielded 0.0, which sat far below every task's floor, so writing to `tests/**` was
strictly worse than doing nothing. After normalisation doing nothing *is* 0.0 - so a gate
still yielding 0.0 would score exactly the same as submitting immediately, and the deterrent
against the #1 reward hack would vanish. Nothing would error and no test would go red; the
gate-rate metric would keep logging hits while the reward quietly stopped caring.

So both ends of the scale are defined once, here, and `GATE_REWARD` is *derived* from the
bottom of it rather than written out again. Changing the formula without the gate constant
is not possible by editing one number, and `rewards/tests/test_compose.py` asserts the
invariant that matters: a gated rollout scores strictly below a do-nothing rollout.
"""

from __future__ import annotations

from collections.abc import Iterable

__all__ = [
    "GATE_REWARD",
    "REWARD_DID_NOTHING",
    "REWARD_MIN",
    "REWARD_SOLVED",
    "compose_reward",
    "normalised_reward",
]

#: The normalised scale, defined once. `REWARD_MIN` is a clamp rather than a natural bound:
#: (p - c) / (n - c) can fall below -1 when a rollout breaks tests that passed untouched,
#: and there is nothing to gain from letting one catastrophic rollout dominate a group's
#: standard deviation.
REWARD_SOLVED = 1.0
REWARD_DID_NOTHING = 0.0
REWARD_MIN = -1.0

#: Derived, not restated: a gated rollout sits at the bottom of the same scale, strictly
#: below `REWARD_DID_NOTHING`. See the module docstring for why this coupling is load-bearing.
GATE_REWARD = REWARD_MIN


def normalised_reward(passed: int, n_tests: int, floor_tests: int = 0) -> float:
    """Fraction of the *movable* tests won, clamped to `[REWARD_MIN, REWARD_SOLVED]`.

    `n_tests` is the count declared in `task.yaml`, not the count that happened to run:
    pinning the denominator to git is what stops shrinking it from paying.
    """
    movable = n_tests - floor_tests
    if movable <= 0:
        raise ValueError(
            f"n_tests ({n_tests}) must exceed floor_tests ({floor_tests}); "
            "a task whose untouched repo passes everything is not failable and "
            "validate_tasks.py rejects it as NOT_FAILABLE"
        )
    raw = (passed - floor_tests) / movable
    return max(REWARD_MIN, min(REWARD_SOLVED, raw))


def compose_reward(
    passed: int,
    n_tests: int,
    floor_tests: int = 0,
    gates: Iterable[str] = (),
) -> float:
    """`reward_v1`: the normalised score, or `GATE_REWARD` if any hard gate fired."""
    if list(gates):
        return GATE_REWARD
    return normalised_reward(passed, n_tests, floor_tests)
