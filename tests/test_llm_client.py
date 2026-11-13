from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import httpx
from openai import APIConnectionError

from openai import APIStatusError
from src.core.retry import RetryPolicy, RetryConfig

from src.core.budget import BudgetConfig, BudgetGuard, BudgetExceededError
from src.core.cost_estimator import CostEstimator
from src.core.key_usage import KeyUsage, KeyUsageError
from src.core.cache import SQLiteCache

from src.core.llm_client import (
    OpenRouterLLMClient,
    LLMClientError,
)


def make_client():
    """Create a client without using a real API key."""
    client = OpenRouterLLMClient(api_key="mock-test-key")
    client.client = MagicMock()
    return client


def test_generate_success():
    client = make_client()

    client.client.chat.completions.create.return_value = (
        SimpleNamespace(
            model="test-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Hello from mock LLM"
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
            ),
        )
    )

    result = client.generate(
        stage="answer_generation",
        messages=[
            {"role": "user", "content": "Hello"}
        ],
        model="test-model",
        temperature=0,
        max_tokens=128,
    )

    assert result.text == "Hello from mock LLM"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20
    assert result.usage.cache_hit is False
    assert result.usage.cost_usd is None


def test_generate_empty_content():
    client = make_client()

    client.client.chat.completions.create.return_value = (
        SimpleNamespace(
            model="test-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None)
                )
            ],
            usage=None,
        )
    )

    with pytest.raises(
        LLMClientError,
        match="no text content"
    ):
        client.generate(
            stage="claim_extraction",
            messages=[
                {"role": "user", "content": "Test"}
            ],
            model="test-model",
            temperature=0,
            max_tokens=128,
        )


def test_generate_api_failure():
    client = make_client()

    client.client.chat.completions.create.side_effect = (
        APIConnectionError(request=MagicMock())
    )

    with pytest.raises(
        LLMClientError,
        match="OpenRouter request failed"
    ):
        client.generate(
            stage="verification",
            messages=[
                {"role": "user", "content": "Test"}
            ],
            model="test-model",
            temperature=0,
            max_tokens=128,
        )


def test_generate_with_cost():
    client = make_client()

    client.client.chat.completions.create.return_value = (
        SimpleNamespace(
            model="test-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Hello"
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                model_extra={"cost": 0.0005},
            ),
        )
    )

    result = client.generate(
        stage="cost_test",
        messages=[
            {"role": "user", "content": "Hello"}
        ],
        model="test-model",
        temperature=0,
        max_tokens=128,
    )

    assert result.usage.cost_usd == 0.0005

    client.client.chat.completions.create.assert_called_once()

    kwargs = (
        client.client.chat.completions.create
        .call_args.kwargs
    )

    assert kwargs["extra_body"] == {
        "usage": {"include": True}
    }


def test_generate_calls_usage_logger():
    mock_logger = MagicMock()

    client = OpenRouterLLMClient(
        api_key="mock-test-key",
        usage_logger=mock_logger,
    )

    client.client = MagicMock()

    client.client.chat.completions.create.return_value = (
        SimpleNamespace(
            model="test-model",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hello")
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                model_extra={"cost": 0.0005},
            ),
        )
    )

    result = client.generate(
        stage="claim_extraction",
        messages=[
            {"role": "user", "content": "Hello"}
        ],
        model="test-model",
        temperature=0,
        max_tokens=128,
    )

    mock_logger.log.assert_called_once_with(result.usage)

    assert result.text == "Hello"
    assert result.usage.cost_usd == 0.0005

def test_budget_blocks_api_call():
    # No API request should be sent when budget is exhausted.
    guard = BudgetGuard(
        BudgetConfig(max_calls=1)
    )

    guard.check_and_reserve(0.005, 15.0)
    guard.settle(0.005, 0.005)

    client = OpenRouterLLMClient(
        api_key="mock-test-key",
        budget_guard=guard,
        cost_estimator=CostEstimator(),
        key_usage_provider=lambda: KeyUsage(
            usage=0.005,
            limit=15.0,
            limit_remaining=14.995,
        ),
    )

    client.client = MagicMock()

    with pytest.raises(BudgetExceededError):
        client.generate(
            stage="test",
            messages=[
                {"role": "user", "content": "Hello"}
            ],
            model="openai/gpt-4o-mini",
            temperature=0,
            max_tokens=128,
        )

    client.client.chat.completions.create.assert_not_called()


def test_budget_settles_success():
    guard = BudgetGuard()

    client = OpenRouterLLMClient(
        api_key="mock-test-key",
        budget_guard=guard,
        cost_estimator=CostEstimator(),
        key_usage_provider=lambda: KeyUsage(
            usage=0.0,
            limit=15.0,
            limit_remaining=15.0,
        ),
    )

    client.client = MagicMock()

    client.client.chat.completions.create.return_value = (
        SimpleNamespace(
            model="openai/gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hello")
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                model_extra={"cost": 0.001},
            ),
        )
    )

    client.generate(
        stage="test",
        messages=[
            {"role": "user", "content": "Hello"}
        ],
        model="openai/gpt-4o-mini",
        temperature=0,
        max_tokens=128,
    )

    assert guard.calls == 1
    assert guard.spent_usd == pytest.approx(0.001)
    assert guard.reserved_usd == pytest.approx(0.0)


def make_budget_client():
    """Create a budget-protected client with mocked dependencies."""
    guard = BudgetGuard()
    mock_logger = MagicMock()

    client = OpenRouterLLMClient(
        api_key="mock-test-key",
        usage_logger=mock_logger,
        budget_guard=guard,
        key_usage_provider=lambda: KeyUsage(
            usage=0.0,
            limit=15.0,
            limit_remaining=15.0,
        ),
    )

    client.client = MagicMock()
    return client, guard, mock_logger


def test_budget_settles_api_failure():
    client, guard, mock_logger = make_budget_client()

    # Simulate a network failure after the request is attempted.
    client.client.chat.completions.create.side_effect = (
        APIConnectionError(request=MagicMock())
    )

    with pytest.raises(LLMClientError):
        client.generate(
            stage="verification",
            messages=[{"role": "user", "content": "Test"}],
            model="openai/gpt-4o-mini",
            temperature=0,
            max_tokens=128,
        )

    assert guard.calls == 1
    assert guard.spent_usd == pytest.approx(0.005)
    assert guard.reserved_usd == pytest.approx(0.0)
    mock_logger.log.assert_not_called()


def test_budget_settles_empty_choices():
    client, guard, mock_logger = make_budget_client()

    client.client.chat.completions.create.return_value = (
        SimpleNamespace(
            model="openai/gpt-4o-mini",
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=0,
                model_extra={"cost": 0.0002},
            ),
        )
    )

    with pytest.raises(
        LLMClientError,
        match="no completion choices"
    ):
        client.generate(
            stage="verification",
            messages=[{"role": "user", "content": "Test"}],
            model="openai/gpt-4o-mini",
            temperature=0,
            max_tokens=128,
        )

    # The API returned usage, so settle the reported cost.
    assert guard.calls == 1
    assert guard.spent_usd == pytest.approx(0.0002)
    assert guard.reserved_usd == pytest.approx(0.0)
    mock_logger.log.assert_not_called()


def test_budget_settles_empty_content():
    client, guard, mock_logger = make_budget_client()

    client.client.chat.completions.create.return_value = (
        SimpleNamespace(
            model="openai/gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None)
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=0,
                model_extra={"cost": 0.0003},
            ),
        )
    )

    with pytest.raises(
        LLMClientError,
        match="no text content"
    ):
        client.generate(
            stage="verification",
            messages=[{"role": "user", "content": "Test"}],
            model="openai/gpt-4o-mini",
            temperature=0,
            max_tokens=128,
        )

    assert guard.calls == 1
    assert guard.spent_usd == pytest.approx(0.0003)
    assert guard.reserved_usd == pytest.approx(0.0)
    mock_logger.log.assert_not_called()


def test_key_usage_failure_blocks_api():
    guard = BudgetGuard()
    mock_logger = MagicMock()

    def failing_key_provider():
        raise KeyUsageError("Mock key usage failure")

    client = OpenRouterLLMClient(
        api_key="mock-test-key",
        usage_logger=mock_logger,
        budget_guard=guard,
        key_usage_provider=failing_key_provider,
    )

    client.client = MagicMock()

    with pytest.raises(KeyUsageError):
        client.generate(
            stage="answer_generation",
            messages=[{"role": "user", "content": "Test"}],
            model="openai/gpt-4o-mini",
            temperature=0,
            max_tokens=128,
        )

    # No reservation or API call should occur.
    assert guard.calls == 0
    assert guard.spent_usd == pytest.approx(0.0)
    assert guard.reserved_usd == pytest.approx(0.0)
    client.client.chat.completions.create.assert_not_called()


def test_unapproved_model_blocks_api():
    client, guard, mock_logger = make_budget_client()

    with pytest.raises(
        ValueError,
        match="No cost reservation policy"
    ):
        client.generate(
            stage="answer_generation",
            messages=[{"role": "user", "content": "Test"}],
            model="unapproved-model",
            temperature=0,
            max_tokens=128,
        )

    assert guard.calls == 0
    assert guard.reserved_usd == pytest.approx(0.0)
    client.client.chat.completions.create.assert_not_called()


def test_invalid_api_cost_uses_reservation():
    client, guard, mock_logger = make_budget_client()

    # Simulate invalid billing data from the API.
    client.client.chat.completions.create.return_value = (
        SimpleNamespace(
            model="openai/gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Hello")
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                model_extra={"cost": "NaN"},
            ),
        )
    )

    result = client.generate(
        stage="verification",
        messages=[{"role": "user", "content": "Test"}],
        model="openai/gpt-4o-mini",
        temperature=0,
        max_tokens=128,
    )

    # Actual cost is unknown, but the budget remains protected.
    assert result.usage.cost_usd is None
    assert guard.spent_usd == pytest.approx(0.005)
    assert guard.reserved_usd == pytest.approx(0.0)
    mock_logger.log.assert_called_once()


def test_api_failure_calls_failure_logger():
    client, guard, mock_logger = make_budget_client()

    client.client.chat.completions.create.side_effect = (
        APIConnectionError(request=MagicMock())
    )

    with pytest.raises(LLMClientError):
        client.generate(
            stage="verification",
            messages=[
                {"role": "user", "content": "Test"}
            ],
            model="openai/gpt-4o-mini",
            temperature=0,
            max_tokens=128,
        )

    # Failure logging must happen exactly once.
    mock_logger.log_failure.assert_called_once()
    mock_logger.log.assert_not_called()

    kwargs = mock_logger.log_failure.call_args.kwargs

    assert kwargs["stage"] == "verification"
    assert kwargs["model"] == "openai/gpt-4o-mini"
    assert kwargs["error_type"] == "LLMClientError"
    assert kwargs["estimated_cost_usd"] == 0.005
    assert kwargs["actual_cost_usd"] is None

    # Budget is conservatively settled.
    assert guard.calls == 1
    assert guard.spent_usd == pytest.approx(0.005)
    assert guard.reserved_usd == pytest.approx(0.0)


def test_empty_response_calls_failure_logger():
    client, guard, mock_logger = make_budget_client()

    client.client.chat.completions.create.return_value = (
        SimpleNamespace(
            model="openai/gpt-4o-mini",
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=0,
                model_extra={"cost": 0.0002},
            ),
        )
    )

    with pytest.raises(LLMClientError):
        client.generate(
            stage="verification",
            messages=[
                {"role": "user", "content": "Test"}
            ],
            model="openai/gpt-4o-mini",
            temperature=0,
            max_tokens=128,
        )

    mock_logger.log_failure.assert_called_once()
    mock_logger.log.assert_not_called()

    kwargs = mock_logger.log_failure.call_args.kwargs

    assert kwargs["actual_cost_usd"] == pytest.approx(0.0002)
    assert guard.spent_usd == pytest.approx(0.0002)


def test_cache_hit_skips_api_and_budget(tmp_path):
    # Prepare a cached response.
    cache = SQLiteCache(
        db_path=str(tmp_path / "cache.db")
    )

    messages = [
        {"role": "user", "content": "Hello"}
    ]

    key = cache.make_key(
        stage="answer_generation",
        model="openai/gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=128,
    )

    cache.set(
        key,
        {
            "text": "Hello from cache",
            "model": "openai/gpt-4o-mini",
        },
    )

    guard = BudgetGuard()
    mock_logger = MagicMock()

    client = OpenRouterLLMClient(
        api_key="mock-test-key",
        usage_logger=mock_logger,
        budget_guard=guard,
        cache=cache,
        key_usage_provider=MagicMock(
            side_effect=AssertionError(
                "Key usage must not be queried on cache hit"
            )
        ),
    )

    client.client = MagicMock()

    result = client.generate(
        stage="answer_generation",
        messages=messages,
        model="openai/gpt-4o-mini",
        temperature=0,
        max_tokens=128,
    )

    assert result.text == "Hello from cache"
    assert result.usage.cache_hit is True
    assert result.usage.cost_usd == 0.0
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0
    assert guard.calls == 0

    client.client.chat.completions.create.assert_not_called()
    mock_logger.log.assert_called_once()


def test_cache_miss_saves_response(tmp_path):
    cache = SQLiteCache(
        db_path=str(tmp_path / "cache.db")
    )

    mock_logger = MagicMock()

    client = OpenRouterLLMClient(
        api_key="mock-test-key",
        usage_logger=mock_logger,
        cache=cache,
    )

    client.client = MagicMock()

    client.client.chat.completions.create.return_value = (
        SimpleNamespace(
            model="openai/gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="New answer"
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=5,
                model_extra={"cost": 0.00001},
            ),
        )
    )

    params = {
        "stage": "answer_generation",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "model": "openai/gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 128,
    }

    first = client.generate(**params)
    second = client.generate(**params)

    assert first.usage.cache_hit is False
    assert second.usage.cache_hit is True
    assert second.text == "New answer"
    assert second.usage.cost_usd == 0.0

    # Only the first request reaches OpenRouter.
    client.client.chat.completions.create.assert_called_once()


def test_failed_api_response_not_cached(tmp_path):
    cache = SQLiteCache(
        db_path=str(tmp_path / "cache.db")
    )

    client = OpenRouterLLMClient(
        api_key="mock-test-key",
        usage_logger=MagicMock(),
        cache=cache,
    )

    client.client = MagicMock()
    client.client.chat.completions.create.return_value = (
        SimpleNamespace(
            model="openai/gpt-4o-mini",
            choices=[],
            usage=None,
        )
    )

    params = {
        "stage": "answer_generation",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "model": "openai/gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 128,
    }

    with pytest.raises(LLMClientError):
        client.generate(**params)

    key = cache.make_key(**params)
    assert cache.get(key) is None


def make_retry_error(status_code):
    request = httpx.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
    )
    response = httpx.Response(
        status_code,
        request=request,
    )
    return APIStatusError(
        "Mock API error",
        response=response,
        body=None,
    )


def make_retry_client(max_attempts=2, budget_guard=None):
    mock_logger = MagicMock()

    client = OpenRouterLLMClient(
        api_key="mock-test-key",
        usage_logger=mock_logger,
        budget_guard=budget_guard,
        key_usage_provider=lambda: KeyUsage(
            usage=0.0,
            limit=15.0,
            limit_remaining=15.0,
        ),
        retry_policy=RetryPolicy(
            RetryConfig(
                max_attempts=max_attempts,
                backoff_seconds=0,
            )
        ),
    )
    client.client = MagicMock()
    return client, mock_logger


def retry_generate(client):
    return client.generate(
        stage="verification",
        messages=[{"role": "user", "content": "Test"}],
        model="openai/gpt-4o-mini",
        temperature=0,
        max_tokens=128,
    )


def test_retry_503_then_success():
    guard = BudgetGuard()
    client, mock_logger = make_retry_client(
        budget_guard=guard
    )

    success = SimpleNamespace(
        model="openai/gpt-4o-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="Recovered")
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            model_extra={"cost": 0.001},
        ),
    )

    # First attempt fails; second succeeds.
    client.client.chat.completions.create.side_effect = [
        make_retry_error(503),
        success,
    ]

    result = retry_generate(client)

    assert result.text == "Recovered"
    assert guard.calls == 2
    assert guard.spent_usd == pytest.approx(0.006)
    assert guard.reserved_usd == pytest.approx(0.0)

    assert (
        client.client.chat.completions.create.call_count == 2
    )
    mock_logger.log_failure.assert_called_once()
    mock_logger.log.assert_called_once()


@pytest.mark.parametrize("status_code", [401, 402])
def test_retry_non_retryable_status(status_code):
    client, mock_logger = make_retry_client()

    client.client.chat.completions.create.side_effect = (
        make_retry_error(status_code)
    )

    with pytest.raises(LLMClientError):
        retry_generate(client)

    # Authentication and payment errors must not retry.
    client.client.chat.completions.create.assert_called_once()
    mock_logger.log_failure.assert_called_once()


def test_retry_exhausted():
    guard = BudgetGuard()
    client, mock_logger = make_retry_client(
        budget_guard=guard
    )

    client.client.chat.completions.create.side_effect = [
        make_retry_error(503),
        make_retry_error(503),
    ]

    with pytest.raises(LLMClientError):
        retry_generate(client)

    assert guard.calls == 2
    assert guard.spent_usd == pytest.approx(0.01)
    assert guard.reserved_usd == pytest.approx(0.0)
    assert (
        client.client.chat.completions.create.call_count == 2
    )
    assert mock_logger.log_failure.call_count == 2


def test_retry_blocked_by_budget():
    guard = BudgetGuard(
        BudgetConfig(max_calls=1)
    )

    client, mock_logger = make_retry_client(
        budget_guard=guard
    )

    client.client.chat.completions.create.side_effect = (
        make_retry_error(503)
    )

    with pytest.raises(BudgetExceededError):
        retry_generate(client)

    # Only the first attempt reaches the API.
    assert guard.calls == 1
    assert guard.spent_usd == pytest.approx(0.005)
    assert guard.reserved_usd == pytest.approx(0.0)
    client.client.chat.completions.create.assert_called_once()
    mock_logger.log_failure.assert_called_once()
