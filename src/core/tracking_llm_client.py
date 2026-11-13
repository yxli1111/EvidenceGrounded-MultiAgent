"""Conservative observation of LLMClient calls.

A generate() call is not necessarily one HTTP request.
Internal retries and failed requests may not be observable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrackingSnapshot:
    generate_calls: int
    completed_calls: int
    observed_cost_usd: float
    uncertain: bool
    reasons: tuple[str, ...]


class TrackingLLMClient:
    """Wrap an LLMClient without changing generate().

    Create a new wrapper per draft.
    Observed cost is not necessarily the final billed amount.
    """

    def __init__(self, client: Any):
        self.client = client
        self.generate_calls = 0
        self.completed_calls = 0
        self.observed_cost_usd = 0.0
        self.uncertain = False
        self._reasons: list[str] = []

    def _mark_uncertain(self, reason: str) -> None:
        self.uncertain = True
        if reason not in self._reasons:
            self._reasons.append(reason)

    def generate(
        self,
        *,
        stage: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ):
        self.generate_calls += 1

        try:
            response = self.client.generate(
                stage=stage,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except BaseException:
            self._mark_uncertain("generate_raised")
            raise

        self.completed_calls += 1

        usage = getattr(response, "usage", None)

        if usage is None:
            self._mark_uncertain("missing_usage")
            return response

        if getattr(usage, "cache_hit", False) is True:
            return response

        raw_cost = getattr(usage, "cost_usd", None)

        if (
            isinstance(raw_cost, bool)
            or not isinstance(raw_cost, (int, float))
        ):
            self._mark_uncertain("missing_cost")

        elif not math.isfinite(raw_cost) or raw_cost < 0:
            self._mark_uncertain("invalid_cost")

        else:
            self.observed_cost_usd += float(raw_cost)

        return response

    def mark_internal_retries_possible(self) -> None:
        self._mark_uncertain(
            "internal_retries_not_observable"
        )

    def snapshot(self) -> TrackingSnapshot:
        return TrackingSnapshot(
            generate_calls=self.generate_calls,
            completed_calls=self.completed_calls,
            observed_cost_usd=self.observed_cost_usd,
            uncertain=self.uncertain,
            reasons=tuple(self._reasons),
        )
