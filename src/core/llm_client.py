import logging
import math
import os
import sqlite3
import time
from typing import Callable

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError, APIStatusError

from contracts.schemas import LLMResponse, LLMUsage
from src.core.budget import BudgetGuard
from src.core.cache import SQLiteCache
from src.core.cost_estimator import CostEstimator
from src.core.key_usage import KeyUsage, get_key_usage
from src.core.retry import RetryPolicy, RetryConfig
from src.core.usage import UsageLogger


logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Base exception for LLM client failures."""
    pass


class LLMAuthenticationError(LLMClientError):
    """Authentication or permission failure."""
    pass


class LLMInsufficientCreditError(LLMClientError):
    """OpenRouter credit or payment failure."""
    pass


class LLMRequestError(LLMClientError):
    """Invalid model, parameters, or request."""
    pass


def classify_api_error(
    error: OpenAIError,
    stage: str,
    attempt: int,
) -> LLMClientError:
    """Convert SDK errors into safe application exceptions."""

    status = None

    if isinstance(error, APIStatusError):
        status = error.status_code

    details = (
        f"OpenRouter request failed at stage '{stage}' "
        f"on attempt {attempt}: {type(error).__name__}"
    )

    if status is not None:
        details += f" (HTTP {status})"

    # Avoid exposing request bodies, headers, or API keys.
    if status in (401, 403):
        return LLMAuthenticationError(details)

    if status == 402:
        return LLMInsufficientCreditError(details)

    if status in (400, 404, 422):
        return LLMRequestError(details)

    return LLMClientError(details)


class OpenRouterLLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        usage_logger: UsageLogger | None = None,
        budget_guard: BudgetGuard | None = None,
        cost_estimator: CostEstimator | None = None,
        key_usage_provider: Callable[[], KeyUsage] | None = None,
        cache: SQLiteCache | None = None,
        retry_policy: RetryPolicy | None = None,
    ):
        load_dotenv()

        key = api_key or os.getenv("OPENROUTER_API_KEY")

        if not key:
            raise LLMClientError(
                "OPENROUTER_API_KEY is not configured."
            )

        # Disable SDK retries to avoid double retrying.
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            max_retries=0,
        )

        self.usage_logger = (
            usage_logger
            if usage_logger is not None
            else UsageLogger()
        )

        self.budget_guard = budget_guard

        self.cost_estimator = (
            cost_estimator
            if cost_estimator is not None
            else CostEstimator()
        )

        self.key_usage_provider = (
            key_usage_provider
            if key_usage_provider is not None
            else lambda: get_key_usage(api_key=key)
        )

        self.cache = cache

        # Default to one attempt for backward compatibility.
        self.retry_policy = (
            retry_policy
            if retry_policy is not None
            else RetryPolicy(
                RetryConfig(max_attempts=1)
            )
        )

    def generate(
        self,
        *,
        stage: str,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:

        # Step 1: Check cache before API and budget.
        cache_key = None

        if self.cache is not None:
            cache_key = self.cache.make_key(
                stage=stage,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            try:
                cached = self.cache.get(cache_key)

            except (OSError, ValueError, sqlite3.Error):
                logger.warning(
                    "Failed to read LLM cache.",
                    exc_info=True,
                )
                cached = None

            if cached is not None:
                try:
                    cached_text = cached["text"]
                    cached_model = cached["model"]

                    if (
                        not isinstance(cached_text, str)
                        or not isinstance(cached_model, str)
                    ):
                        raise ValueError(
                            "Invalid cached response fields."
                        )

                    result = LLMResponse(
                        text=cached_text,
                        usage=LLMUsage(
                            stage=stage,
                            model=cached_model,
                            input_tokens=0,
                            output_tokens=0,
                            cost_usd=0.0,
                            latency_seconds=0.0,
                            cache_hit=True,
                        ),
                    )

                except (KeyError, TypeError, ValueError):
                    logger.warning(
                        "Invalid cached response; ignoring cache."
                    )

                else:
                    # Cache hits skip API calls and budget checks.
                    try:
                        self.usage_logger.log(result.usage)

                    except OSError:
                        logger.warning(
                            "Failed to log cache hit.",
                            exc_info=True,
                        )

                    return result

        # Step 2: Start bounded retry attempts.
        max_attempts = self.retry_policy.config.max_attempts

        for attempt in range(1, max_attempts + 1):
            estimated_cost = None
            actual_cost = None
            reserved = False
            request_attempted = False
            api_error = None

            # Step 3: Reserve budget for each real attempt.
            if self.budget_guard is not None:
                estimated_cost = self.cost_estimator.estimate(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )

                key_info = self.key_usage_provider()

                self.budget_guard.check_and_reserve(
                    estimated_cost_usd=estimated_cost,
                    key_remaining_usd=key_info.limit_remaining,
                )

                reserved = True

            start_time = time.perf_counter()

            try:
                # Step 4: Make one OpenRouter API request.
                request_attempted = True

                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        extra_body={
                            "usage": {"include": True}
                        },
                    )

                except OpenAIError as exc:
                    # Preserve SDK error for retry decisions.
                    api_error = exc

                    application_error = classify_api_error(
                        error=exc,
                        stage=stage,
                        attempt=attempt,
                    )

                    raise application_error from exc

                # Step 5: Parse actual billing information.
                usage = response.usage

                if usage is not None:
                    extra = getattr(
                        usage,
                        "model_extra",
                        None,
                    ) or {}

                    raw_cost = extra.get("cost")

                    if raw_cost is not None:
                        try:
                            parsed_cost = float(raw_cost)

                        except (TypeError, ValueError):
                            parsed_cost = None

                        if (
                            parsed_cost is not None
                            and math.isfinite(parsed_cost)
                            and parsed_cost >= 0
                        ):
                            actual_cost = parsed_cost

                        else:
                            logger.warning(
                                "Invalid OpenRouter cost; "
                                "using budget estimate."
                            )

                # Step 6: Validate API response content.
                if not response.choices:
                    raise LLMClientError(
                        "OpenRouter returned no completion choices."
                    )

                content = response.choices[0].message.content

                if content is None:
                    raise LLMClientError(
                        "OpenRouter returned no text content."
                    )

                # Step 7: Build the public response object.
                result = LLMResponse(
                    text=content,
                    usage=LLMUsage(
                        stage=stage,
                        model=response.model or model,
                        input_tokens=(
                            usage.prompt_tokens
                            if usage is not None
                            else None
                        ),
                        output_tokens=(
                            usage.completion_tokens
                            if usage is not None
                            else None
                        ),
                        cost_usd=actual_cost,
                        latency_seconds=(
                            time.perf_counter() - start_time
                        ),
                        cache_hit=False,
                    ),
                )

            except Exception as exc:
                # Step 8: Log each failed API attempt.
                if request_attempted:
                    try:
                        self.usage_logger.log_failure(
                            stage=stage,
                            model=model,
                            error_type=type(exc).__name__,
                            estimated_cost_usd=estimated_cost,
                            actual_cost_usd=actual_cost,
                            latency_seconds=(
                                time.perf_counter() - start_time
                            ),
                        )

                    except OSError:
                        logger.warning(
                            "Failed to write LLM failure log.",
                            exc_info=True,
                        )

                # Only retry eligible OpenAI SDK errors.
                if api_error is None:
                    raise

                if not self.retry_policy.should_retry(
                    api_error,
                    attempt,
                ):
                    raise

            finally:
                # Step 9: Settle budget after every attempt.
                if reserved:
                    self.budget_guard.settle(
                        estimated_cost_usd=estimated_cost,
                        actual_cost_usd=actual_cost,
                    )

            # Step 10: Apply Retry-After or fallback delay.
            if api_error is not None:
                delay = self.retry_policy.get_delay(
                    api_error
                )

                logger.warning(
                    "Retrying OpenRouter request: "
                    "stage=%s, attempt=%d, delay=%.2fs",
                    stage,
                    attempt,
                    delay,
                )

                if delay > 0:
                    time.sleep(delay)

                continue

            # Step 11: Log successful API usage.
            try:
                self.usage_logger.log(result.usage)

            except OSError:
                logger.warning(
                    "Failed to write LLM usage log.",
                    exc_info=True,
                )

            # Step 12: Cache successful responses only.
            if self.cache is not None and cache_key is not None:
                try:
                    self.cache.set(
                        cache_key,
                        {
                            "text": result.text,
                            "model": result.usage.model,
                        },
                    )

                except (OSError, ValueError, sqlite3.Error):
                    logger.warning(
                        "Failed to write LLM cache.",
                        exc_info=True,
                    )

            return result

        # Defensive fallback.
        raise LLMClientError(
            "OpenRouter request exhausted all attempts."
        )
