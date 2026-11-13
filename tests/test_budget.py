import math
import pytest

from src.core.budget import (
    BudgetConfig,
    BudgetGuard,
    BudgetExceededError,
)


def test_budget_allows_request():
    guard = BudgetGuard()

    guard.check_and_reserve(
        estimated_cost_usd=0.01,
        key_remaining_usd=15.0,
    )

    assert guard.calls == 1
    assert guard.reserved_usd == pytest.approx(0.01)

    guard.settle(
        estimated_cost_usd=0.01,
        actual_cost_usd=0.008,
    )

    assert guard.spent_usd == pytest.approx(0.008)
    assert guard.reserved_usd == pytest.approx(0.0)


def test_budget_rejects_call_limit():
    guard = BudgetGuard(
        BudgetConfig(max_calls=2)
    )

    for _ in range(2):
        guard.check_and_reserve(0.001, 15.0)
        guard.settle(0.001, 0.001)

    with pytest.raises(
        BudgetExceededError,
        match="Maximum API call count"
    ):
        guard.check_and_reserve(0.001, 15.0)


def test_budget_rejects_cost_limit():
    guard = BudgetGuard(
        BudgetConfig(max_cost_usd=0.10)
    )

    guard.check_and_reserve(0.08, 15.0)
    guard.settle(0.08, 0.08)

    with pytest.raises(
        BudgetExceededError,
        match="Run budget"
    ):
        guard.check_and_reserve(0.03, 15.0)


def test_budget_rejects_low_key_balance():
    guard = BudgetGuard()

    with pytest.raises(
        BudgetExceededError,
        match="Key remaining limit"
    ):
        guard.check_and_reserve(
            estimated_cost_usd=0.01,
            key_remaining_usd=10.005,
        )


def test_budget_rejects_unknown_balance():
    guard = BudgetGuard()

    with pytest.raises(
        BudgetExceededError,
        match="unknown"
    ):
        guard.check_and_reserve(
            estimated_cost_usd=0.01,
            key_remaining_usd=None,
        )


def test_unknown_actual_cost_uses_estimate():
    guard = BudgetGuard()

    guard.check_and_reserve(0.01, 15.0)
    guard.settle(0.01, None)

    assert guard.spent_usd == pytest.approx(0.01)
    assert guard.reserved_usd == pytest.approx(0.0)


def test_budget_rejects_nan_estimate():
    guard = BudgetGuard()

    with pytest.raises(ValueError):
        guard.check_and_reserve(
            estimated_cost_usd=math.nan,
            key_remaining_usd=15.0,
        )

    assert guard.calls == 0
    assert guard.reserved_usd == 0.0


def test_budget_rejects_invalid_key_remaining():
    guard = BudgetGuard()

    with pytest.raises(BudgetExceededError):
        guard.check_and_reserve(
            estimated_cost_usd=0.005,
            key_remaining_usd=math.inf,
        )

    assert guard.calls == 0


def test_settle_invalid_cost_preserves_state():
    guard = BudgetGuard()

    guard.check_and_reserve(0.005, 15.0)

    with pytest.raises(ValueError):
        guard.settle(
            estimated_cost_usd=0.005,
            actual_cost_usd=math.nan,
        )

    # Invalid settlement must not corrupt the budget.
    assert guard.calls == 1
    assert guard.reserved_usd == pytest.approx(0.005)
    assert guard.spent_usd == pytest.approx(0.0)


def test_settle_without_reservation_rejected():
    guard = BudgetGuard()

    with pytest.raises(ValueError):
        guard.settle(
            estimated_cost_usd=0.005,
            actual_cost_usd=0.001,
        )

    assert guard.spent_usd == pytest.approx(0.0)
