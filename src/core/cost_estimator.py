from dataclasses import dataclass


@dataclass(frozen=True)
class CostEstimateConfig:
    # Fixed reservation for approved development models only.
    default_reservation_usd: float = 0.005
    allowed_models: tuple[str, ...] = (
        "openai/gpt-4o-mini",
    )


class CostEstimator:
    """Estimate API cost for development-time budget checks."""

    def __init__(
        self,
        config: CostEstimateConfig | None = None,
    ):
        self.config = (
            config if config is not None else CostEstimateConfig()
        )

        if self.config.default_reservation_usd <= 0:
            raise ValueError("Reservation must be positive")

    def estimate(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> float:
        # Validate request parameters.
        if not model:
            raise ValueError("Model ID cannot be empty")

        if not messages:
            raise ValueError("Messages cannot be empty")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        # Reject models without an approved reservation policy.
        if model not in self.config.allowed_models:
            raise ValueError(
                f"No cost reservation policy for model: {model}"
            )

        return self.config.default_reservation_usd
