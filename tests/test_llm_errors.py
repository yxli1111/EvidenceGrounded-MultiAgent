from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIStatusError

from src.core.llm_client import (
    OpenRouterLLMClient,
    LLMClientError,
    LLMAuthenticationError,
    LLMInsufficientCreditError,
    LLMRequestError,
)
from src.core.retry import RetryConfig, RetryPolicy


def make_http_error(status_code, headers=None):
    request = httpx.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
    )
    response = httpx.Response(
        status_code,
        request=request,
        headers=headers or {},
    )
    return APIStatusError(
        "Mock HTTP error",
        response=response,
        body=None,
    )


def make_client(max_attempts=1):
    client = OpenRouterLLMClient(
        api_key="mock-test-key",
        usage_logger=MagicMock(),
        retry_policy=RetryPolicy(
            RetryConfig(
                max_attempts=max_attempts,
                backoff_seconds=0,
            )
        ),
    )
    client.client = MagicMock()
    return client


def generate(client):
    return client.generate(
        stage="verification",
        messages=[
            {"role": "user", "content": "Test"}
        ],
        model="openai/gpt-4o-mini",
        temperature=0,
        max_tokens=128,
    )


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_errors(status):
    client = make_client()
    client.client.chat.completions.create.side_effect = (
        make_http_error(status)
    )

    with pytest.raises(LLMAuthenticationError) as exc:
        generate(client)

    assert isinstance(exc.value, LLMClientError)
    client.client.chat.completions.create.assert_called_once()


def test_insufficient_credit_error():
    client = make_client()
    client.client.chat.completions.create.side_effect = (
        make_http_error(402)
    )

    with pytest.raises(LLMInsufficientCreditError):
        generate(client)

    client.client.chat.completions.create.assert_called_once()


@pytest.mark.parametrize("status", [400, 404, 422])
def test_invalid_request_errors(status):
    client = make_client()
    client.client.chat.completions.create.side_effect = (
        make_http_error(status)
    )

    with pytest.raises(LLMRequestError):
        generate(client)

    client.client.chat.completions.create.assert_called_once()


def test_retry_after_used_by_llm_client():
    client = make_client(max_attempts=2)

    client.client.chat.completions.create.side_effect = [
        make_http_error(
            429,
            headers={"Retry-After": "2"},
        ),
        make_http_error(401),
    ]

    # Mock sleep to avoid waiting during tests.
    with patch(
        "src.core.llm_client.time.sleep"
    ) as mock_sleep:
        with pytest.raises(LLMAuthenticationError):
            generate(client)

    mock_sleep.assert_called_once_with(2.0)
    assert (
        client.client.chat.completions.create.call_count == 2
    )


def test_retry_exhaustion_preserves_base_error():
    client = make_client(max_attempts=2)

    client.client.chat.completions.create.side_effect = [
        make_http_error(503),
        make_http_error(503),
    ]

    with pytest.raises(LLMClientError):
        generate(client)

    assert (
        client.client.chat.completions.create.call_count == 2
    )
