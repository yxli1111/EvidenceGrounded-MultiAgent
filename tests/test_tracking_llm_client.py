"""Offline tests for TrackingLLMClient."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core.tracking_llm_client import TrackingLLMClient


ARGS = {
    "stage": "claim_extraction",
    "messages": [
        {"role": "user", "content": "test"}
    ],
    "model": "test",
    "temperature": 0.0,
    "max_tokens": 64,
}


def result(cost=None, cache_hit=False):
    return SimpleNamespace(
        text="ok",
        usage=SimpleNamespace(
            cost_usd=cost,
            cache_hit=cache_hit,
        ),
    )


def test_success_tracks_observed_cost():
    client = Mock()
    client.generate.side_effect = [
        result(0.001),
        result(0.002),
    ]

    tracker = TrackingLLMClient(client)

    tracker.generate(**ARGS)
    tracker.generate(**ARGS)

    snapshot = tracker.snapshot()

    assert snapshot.generate_calls == 2
    assert snapshot.completed_calls == 2
    assert snapshot.observed_cost_usd == pytest.approx(0.003)
    assert not snapshot.uncertain


def test_missing_cost_is_uncertain():
    client = Mock(
        generate=Mock(return_value=result())
    )
    tracker = TrackingLLMClient(client)

    tracker.generate(**ARGS)

    assert tracker.snapshot().uncertain
    assert "missing_cost" in tracker.snapshot().reasons


def test_failure_remains_uncertain():
    client = Mock(
        generate=Mock(
            side_effect=RuntimeError("network")
        )
    )
    tracker = TrackingLLMClient(client)

    with pytest.raises(RuntimeError):
        tracker.generate(**ARGS)

    assert tracker.snapshot().generate_calls == 1
    assert tracker.snapshot().completed_calls == 0
    assert tracker.snapshot().uncertain


def test_cache_hit_does_not_count_cost():
    client = Mock(
        generate=Mock(
            return_value=result(cache_hit=True)
        )
    )
    tracker = TrackingLLMClient(client)

    tracker.generate(**ARGS)

    assert tracker.snapshot().observed_cost_usd == 0
    assert not tracker.snapshot().uncertain


def test_invalid_cost_is_uncertain():
    client = Mock(
        generate=Mock(
            return_value=result(float("nan"))
        )
    )
    tracker = TrackingLLMClient(client)

    tracker.generate(**ARGS)

    assert tracker.snapshot().uncertain


def test_internal_retry_uncertainty():
    tracker = TrackingLLMClient(Mock())

    tracker.mark_internal_retries_possible()

    assert (
        "internal_retries_not_observable"
        in tracker.snapshot().reasons
    )
