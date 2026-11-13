from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from openai import APIStatusError

from src.core.budget import BudgetGuard
from src.core.cache import SQLiteCache
from src.core.key_usage import KeyUsage
from src.core.llm_client import OpenRouterLLMClient
from src.core.retry import RetryConfig, RetryPolicy


MODEL = "openai/gpt-4o-mini"

MESSAGES = [
    {"role": "user", "content": "Hello"}
]


def make_error(status_code: int):
    """Create a simulated OpenRouter HTTP error."""
    request = httpx.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
    )

    response = httpx.Response(
        status_code=status_code,
        request=request,
    )

    return APIStatusError(
        "Mock API error",
        response=response,
        body=None,
    )


def make_success_response():
    """Create a simulated successful LLM response."""
    return SimpleNamespace(
        model=MODEL,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello from OpenRouter"
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            model_extra={"cost": 0.001},
        ),
    )


def make_test_client(cache, guard, logger):
    """Create a client with mocked external dependencies."""
    client = OpenRouterLLMClient(
        api_key="mock-test-key",
        usage_logger=logger,
        budget_guard=guard,
        cache=cache,
        key_usage_provider=lambda: KeyUsage(
            usage=0.0,
            limit=15.0,
            limit_remaining=15.0,
        ),
        retry_policy=RetryPolicy(
            RetryConfig(
                max_attempts=2,
                backoff_seconds=0,
            )
        ),
    )

    # Prevent real OpenRouter requests.
    client.client = MagicMock()

    return client


def generate(client):
    """Send a standard test request."""
    return client.generate(
        stage="verification",
        messages=MESSAGES,
        model=MODEL,
        temperature=0,
        max_tokens=128,
    )


def test_cache_hit_skips_retry_and_budget(tmp_path):
    cache = SQLiteCache(
        db_path=str(tmp_path / "cache.db")
    )

    key = cache.make_key(
        stage="verification",
        model=MODEL,
        messages=MESSAGES,
        temperature=0,
        max_tokens=128,
    )

    cache.set(
        key,
        {
            "text": "Cached answer",
            "model": MODEL,
        },
    )

    guard = BudgetGuard()
    mock_logger = MagicMock()

    client = make_test_client(
        cache,
        guard,
        mock_logger,
    )

    result = generate(client)

    # Cache hit must bypass OpenRouter and BudgetGuard.
    assert result.text == "Cached answer"
    assert result.usage.cache_hit is True
    assert result.usage.cost_usd == 0.0

    assert guard.calls == 0
    assert guard.spent_usd == pytest.approx(0.0)
    assert guard.reserved_usd == pytest.approx(0.0)

    client.client.chat.completions.create.assert_not_called()
    mock_logger.log_failure.assert_not_called()
    mock_logger.log.assert_called_once()


def test_retry_success_then_cache_hit(tmp_path):
    cache = SQLiteCache(
        db_path=str(tmp_path / "cache.db")
    )

    guard = BudgetGuard()
    mock_logger = MagicMock()

    client = make_test_client(
        cache,
        guard,
        mock_logger,
    )

    # First attempt fails; second succeeds.
    client.client.chat.completions.create.side_effect = [
        make_error(503),
        make_success_response(),
    ]

    first_result = generate(client)

    assert first_result.text == "Hello from OpenRouter"
    assert first_result.usage.cache_hit is False

    # Two real attempts, each with its own budget settlement.
    assert guard.calls == 2
    assert guard.spent_usd == pytest.approx(0.006)
    assert guard.reserved_usd == pytest.approx(0.0)

    assert (
        client.client.chat.completions.create.call_count == 2
    )

    # One failed attempt and one successful attempt.
    mock_logger.log_failure.assert_called_once()
    mock_logger.log.assert_called_once()

    # The next identical request should hit the cache.
    second_result = generate(client)

    assert second_result.text == "Hello from OpenRouter"
    assert second_result.usage.cache_hit is True
    assert second_result.usage.cost_usd == 0.0

    # Cache hit must not create another API attempt.
    assert guard.calls == 2
    assert guard.spent_usd == pytest.approx(0.006)
    assert (
        client.client.chat.completions.create.call_count == 2
    )

    mock_logger.log_failure.assert_called_once()
    assert mock_logger.log.call_count == 2
