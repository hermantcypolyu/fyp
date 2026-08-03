"""Unit tests for the transport-failure sentinel.

The load-bearing test is `test_a_non_200_never_yields_a_numeric_reward`. Under ADR 0006
`0.0` is a legitimate score - it is what a do-nothing rollout earns - so a `/verify` failure
that returns `0.0` puts a counterfeit do-nothing sample into the group, shifts the mean and
standard deviation GRPO normalises against, and reports nothing. Both upstream reference
implementations do exactly that (`resp.json().get("reward", 0.0) if status_code == 200 else
0.0`), which is why this is pinned rather than assumed.
"""

from __future__ import annotations

import pytest

from rewards.transport import (
    DiscardRateTooHigh,
    DiscardTracker,
    VerifyTransportError,
    reward_from_verify_response,
    score_or_discard,
)

# --- the invariant: a broken response is never a score ----------------------------------

BROKEN_RESPONSES = [
    pytest.param(500, {"reward": 0.87}, id="500-with-a-reward-body"),
    pytest.param(503, {}, id="503-empty"),
    pytest.param(404, None, id="404-no-body"),
    pytest.param(200, None, id="200-but-null-body"),
    pytest.param(200, {}, id="200-but-no-reward-key"),
    pytest.param(200, {"error": "boom"}, id="200-but-an-error-body"),
    pytest.param(200, {"reward": None}, id="200-but-reward-is-null"),
    pytest.param(200, {"reward": "0.5"}, id="200-but-reward-is-a-string"),
    pytest.param(200, {"reward": True}, id="200-but-reward-is-a-bool"),
    pytest.param(200, {"reward": float("nan")}, id="200-but-reward-is-nan"),
    pytest.param(200, {"reward": float("inf")}, id="200-but-reward-is-inf"),
    pytest.param(200, [0.5], id="200-but-body-is-a-list"),
]


@pytest.mark.parametrize(("status_code", "payload"), BROKEN_RESPONSES)
def test_a_non_200_never_yields_a_numeric_reward(status_code, payload):
    with pytest.raises(VerifyTransportError):
        reward_from_verify_response(status_code=status_code, payload=payload)


@pytest.mark.parametrize(("status_code", "payload"), BROKEN_RESPONSES)
def test_a_broken_response_discards_rather_than_scoring(status_code, payload):
    result = score_or_discard(status_code=status_code, payload=payload)
    assert result is None
    # the point of the whole module: not zero, not any number
    assert not isinstance(result, (int, float))


def test_the_upstream_zero_default_is_not_reproduced():
    """A 500 carrying a reward body must not be read as a score of any kind."""
    with pytest.raises(VerifyTransportError, match="unscorable"):
        reward_from_verify_response(status_code=500, payload={"reward": 0.0})


# --- the happy path still works ----------------------------------------------------------


@pytest.mark.parametrize("reward", [1.0, 0.0, -1.0, 0.5, -0.25, 0, 1])
def test_a_valid_response_returns_the_reward_including_zero_and_negative(reward):
    got = reward_from_verify_response(status_code=200, payload={"reward": reward})
    assert got == pytest.approx(float(reward))
    assert isinstance(got, float)


def test_a_genuine_zero_is_a_score_not_a_discard():
    """0.0 from a healthy verifier means 'achieved nothing' and must be kept."""
    assert score_or_discard(status_code=200, payload={"reward": 0.0}) == 0.0


def test_a_gate_of_minus_one_survives_the_transport_layer():
    assert score_or_discard(status_code=200, payload={"reward": -1.0}) == -1.0


# --- discard accounting ------------------------------------------------------------------


def test_the_tracker_counts_scored_and_discarded_rollouts():
    tracker = DiscardTracker()
    score_or_discard(status_code=200, payload={"reward": 0.5}, tracker=tracker)
    score_or_discard(status_code=500, payload={}, tracker=tracker)
    assert tracker.scored == 1
    assert tracker.discarded == 1
    assert tracker.rate == pytest.approx(0.5)
    assert tracker.summary() == {"rollouts_scored": 1, "rollouts_discarded": 1, "discard_rate": 0.5}


def test_a_healthy_run_does_not_trip_the_threshold():
    tracker = DiscardTracker(max_discard_rate=0.02, min_samples=20)
    for _ in range(100):
        score_or_discard(status_code=200, payload={"reward": 0.5}, tracker=tracker)
    tracker.check()  # must not raise


def test_a_small_sample_does_not_trip_the_threshold():
    """Two failures out of three is 67%, but three rollouts is not evidence of anything."""
    tracker = DiscardTracker(max_discard_rate=0.02, min_samples=20)
    for status in (200, 500, 500):
        score_or_discard(status_code=status, payload={"reward": 0.5}, tracker=tracker)
    tracker.check()  # must not raise


def test_a_dead_environment_fails_loudly():
    tracker = DiscardTracker(max_discard_rate=0.02, min_samples=20)
    for _ in range(50):
        score_or_discard(status_code=503, payload={}, tracker=tracker)
    with pytest.raises(DiscardRateTooHigh, match="unscorable"):
        tracker.check()


def test_the_failure_message_carries_recent_reasons():
    tracker = DiscardTracker(max_discard_rate=0.02, min_samples=5)
    for _ in range(10):
        score_or_discard(status_code=503, payload={}, tracker=tracker, task_id="bugfix_probe")
    with pytest.raises(DiscardRateTooHigh) as exc:
        tracker.check()
    assert "HTTP 503" in str(exc.value)
    assert "bugfix_probe" in str(exc.value)


def test_reset_clears_the_window():
    tracker = DiscardTracker()
    score_or_discard(status_code=503, payload={}, tracker=tracker)
    tracker.reset()
    assert tracker.total == 0
    assert tracker.rate == 0.0
