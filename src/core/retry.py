import math
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
)


@dataclass(frozen=True)
class RetryConfig:
    # Total attempts include the initial request.
    max_attempts: int = 2
    backoff_seconds: float = 0.5
    max_retry_after_seconds: float = 30.0


class RetryPolicy:
    """Determine retry eligibility and waiting time."""

    RETRYABLE_STATUS_CODES = {
        429, 500, 502, 503, 504
    }

    def __init__(
        self,
        config: RetryConfig | None = None,
    ):
        self.config = (
            config if config is not None else RetryConfig()
        )

        if self.config.max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1"
            )

        if (
            not math.isfinite(self.config.backoff_seconds)
            or self.config.backoff_seconds < 0
        ):
            raise ValueError(
                "backoff_seconds must be finite and non-negative"
            )

        if (
            not math.isfinite(
                self.config.max_retry_after_seconds
            )
            or self.config.max_retry_after_seconds < 0
        ):
            raise ValueError(
                "max_retry_after_seconds must be finite "
                "and non-negative"
            )

    def is_retryable(self, error: Exception) -> bool:
        """Check whether the error is transient."""

        if isinstance(
            error,
            (APIConnectionError, APITimeoutError),
        ):
            return True

        if isinstance(error, APIStatusError):
            return (
                error.status_code
                in self.RETRYABLE_STATUS_CODES
            )

        return False

    def should_retry(
        self,
        error: Exception,
        attempt: int,
    ) -> bool:
        """Check retry eligibility and attempt limit."""

        if attempt < 1:
            raise ValueError(
                "Attempt number must be positive"
            )

        return (
            attempt < self.config.max_attempts
            and self.is_retryable(error)
        )

    def get_delay(self, error: Exception) -> float:
        """Respect Retry-After when valid and available."""

        fallback = self.config.backoff_seconds

        if not isinstance(error, APIStatusError):
            return fallback

        header = error.response.headers.get(
            "retry-after"
        )

        if not header:
            return fallback

        # Retry-After can be seconds or an HTTP date.
        try:
            seconds = float(header)

            if not math.isfinite(seconds):
                return fallback

        except ValueError:
            try:
                retry_date = parsedate_to_datetime(header)

                if retry_date.tzinfo is None:
                    retry_date = retry_date.replace(
                        tzinfo=timezone.utc
                    )

                seconds = (
                    retry_date
                    - datetime.now(timezone.utc)
                ).total_seconds()

            except (TypeError, ValueError, OverflowError):
                return fallback

        if seconds < 0:
            return fallback

        # Bound the waiting time to avoid excessive delays.
        return min(
            seconds,
            self.config.max_retry_after_seconds,
        )
