import math
from dataclasses import dataclass


class BudgetExceededError(Exception):
    """Raised when an API request violates budget limits."""
    pass


@dataclass
class BudgetConfig:
    max_calls: int = 20
    max_cost_usd: float = 0.10
    min_key_remaining_usd: float = 10.00


class BudgetGuard:
    """Track and limit API requests within one run."""

    def __init__(self, config: BudgetConfig | None = None):
        self.config = (
            config if config is not None else BudgetConfig()
        )

        # Validate budget configuration.
        if self.config.max_calls < 1:
            raise ValueError("max_calls must be positive")

        if (
            not math.isfinite(self.config.max_cost_usd)
            or self.config.max_cost_usd <= 0
        ):
            raise ValueError("max_cost_usd must be positive and finite")

        if (
            not math.isfinite(self.config.min_key_remaining_usd)
            or self.config.min_key_remaining_usd < 0
        ):
            raise ValueError(
                "min_key_remaining_usd must be non-negative and finite"
            )

        self.calls = 0
        self.spent_usd = 0.0
        self.reserved_usd = 0.0

    def check_and_reserve(
        self,
        estimated_cost_usd: float,
        key_remaining_usd: float | None,
    ) -> None:
        """Check limits and reserve budget before an API call."""

        # Step 1: Validate estimated cost.
        if (
            not math.isfinite(estimated_cost_usd)
            or estimated_cost_usd <= 0
        ):
            raise ValueError("Estimated cost must be positive and finite")

        # Step 2: Check maximum API call count.
        if self.calls >= self.config.max_calls:
            raise BudgetExceededError(
                "Maximum API call count reached"
            )

        # Step 3: Check run-level budget.
        projected_cost = (
            self.spent_usd
            + self.reserved_usd
            + estimated_cost_usd
        )

        if projected_cost > self.config.max_cost_usd:
            raise BudgetExceededError(
                "Run budget would be exceeded"
            )

        # Step 4: Check key remaining limit.
        if key_remaining_usd is None:
            raise BudgetExceededError(
                "Key remaining limit is unknown"
            )

        # Reject invalid key remaining values.
        if (
            not math.isfinite(key_remaining_usd)
            or key_remaining_usd < 0
        ):
            raise BudgetExceededError(
                "Invalid key remaining limit"
            )

        if (
            key_remaining_usd
            - self.reserved_usd
            - estimated_cost_usd
            < self.config.min_key_remaining_usd
        ):
            raise BudgetExceededError(
                "Key remaining limit is too low"
            )

        # Step 5: Reserve budget for the approved request.
        self.calls += 1
        self.reserved_usd += estimated_cost_usd

    def settle(
        self,
        estimated_cost_usd: float,
        actual_cost_usd: float | None,
    ) -> None:
        """Reconcile reserved budget after an API call."""

        # Step 1: Validate costs before modifying state.
        if (
            not math.isfinite(estimated_cost_usd)
            or estimated_cost_usd <= 0
        ):
            raise ValueError("Invalid estimated cost")

        if actual_cost_usd is not None:
            if (
                not math.isfinite(actual_cost_usd)
                or actual_cost_usd < 0
            ):
                raise ValueError("Invalid actual cost")

        # Prevent settlement without a matching reservation amount.
        if self.reserved_usd + 1e-12 < estimated_cost_usd:
            raise ValueError("Insufficient reserved budget")

        # Step 2: Use estimate when actual billing is unknown.
        settled_cost = (
            estimated_cost_usd
            if actual_cost_usd is None
            else actual_cost_usd
        )

        # Step 3: Release reservation and record settled cost.
        self.reserved_usd = max(
            0.0,
            self.reserved_usd - estimated_cost_usd,
        )

        self.spent_usd += settled_cost
