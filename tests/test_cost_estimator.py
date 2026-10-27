import pytest

from src.core.cost_estimator import (
    CostEstimator,
    CostEstimateConfig,
)


def test_default_reservation():
    estimator = CostEstimator()

    cost = estimator.estimate(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Hello"}
        ],
        max_tokens=128,
    )

    assert cost == pytest.approx(0.005)


def test_custom_reservation():
    estimator = CostEstimator(
        CostEstimateConfig(
            default_reservation_usd=0.01,
            allowed_models=("test-model",),
        )
    )

    cost = estimator.estimate(
        model="test-model",
        messages=[
            {"role": "user", "content": "Test"}
        ],
        max_tokens=512,
    )

    assert cost == pytest.approx(0.01)


def test_invalid_reservation():
    with pytest.raises(ValueError):
        CostEstimator(
            CostEstimateConfig(
                default_reservation_usd=0
            )
        )


def test_invalid_max_tokens():
    estimator = CostEstimator()

    with pytest.raises(
        ValueError,
        match="max_tokens"
    ):
        estimator.estimate(
            model="test-model",
            messages=[
                {"role": "user", "content": "Test"}
            ],
            max_tokens=0,
        )

def test_rejects_unapproved_model():
    estimator = CostEstimator()

    with pytest.raises(
        ValueError,
        match="No cost reservation policy"
    ):
        estimator.estimate(
            model="unapproved-model",
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=128,
        )

def test_rejects_unapproved_model():
    estimator = CostEstimator()

    with pytest.raises(
        ValueError,
        match="No cost reservation policy"
    ):
        estimator.estimate(
            model="unapproved-model",
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            max_tokens=128,
        )
