import httpx
import pytest

from openai import APIConnectionError, APIStatusError

from src.core.retry import RetryConfig, RetryPolicy


def make_status_error(status_code: int):
    """Create a simulated OpenAI-compatible HTTP error."""
    request = httpx.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
    )

    response = httpx.Response(
        status_code=status_code,
        request=request,
    )

    return APIStatusError(
        "Mock HTTP error",
        response=response,
        body=None,
    )


@pytest.mark.parametrize(
    "status_code",
    [429, 500, 502, 503, 504],
)
def test_retryable_http_status(status_code):
    policy = RetryPolicy()
    error = make_status_error(status_code)

    assert policy.should_retry(error, attempt=1)


@pytest.mark.parametrize(
    "status_code",
    [400, 401, 402, 403, 404],
)
def test_non_retryable_http_status(status_code):
    policy = RetryPolicy()
    error = make_status_error(status_code)

    assert not policy.should_retry(error, attempt=1)


def test_network_error_is_retryable():
    policy = RetryPolicy()

    request = httpx.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
    )

    error = APIConnectionError(request=request)

    assert policy.should_retry(error, attempt=1)


def test_retry_stops_after_max_attempts():
    policy = RetryPolicy(
        RetryConfig(max_attempts=2)
    )

    error = make_status_error(429)

    assert policy.should_retry(error, attempt=1)
    assert not policy.should_retry(error, attempt=2)


def test_retry_disabled():
    policy = RetryPolicy(
        RetryConfig(max_attempts=1)
    )

    error = make_status_error(503)

    assert not policy.should_retry(error, attempt=1)


def test_invalid_retry_config():
    with pytest.raises(ValueError):
        RetryPolicy(
            RetryConfig(max_attempts=0)
        )

    with pytest.raises(ValueError):
        RetryPolicy(
            RetryConfig(backoff_seconds=-1)
        )


def test_unknown_error_not_retryable():
    policy = RetryPolicy()

    error = RuntimeError("Unexpected failure")

    assert not policy.should_retry(error, attempt=1)


def test_retry_after_seconds():
    policy = RetryPolicy(
        RetryConfig(
            backoff_seconds=0.5,
            max_retry_after_seconds=30,
        )
    )

    request = httpx.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
    )

    response = httpx.Response(
        429,
        headers={"Retry-After": "3"},
        request=request,
    )

    error = APIStatusError(
        "Rate limited",
        response=response,
        body=None,
    )

    assert policy.get_delay(error) == 3.0


def test_retry_after_is_capped():
    policy = RetryPolicy(
        RetryConfig(max_retry_after_seconds=30)
    )

    request = httpx.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
    )

    response = httpx.Response(
        429,
        headers={"Retry-After": "120"},
        request=request,
    )

    error = APIStatusError(
        "Rate limited",
        response=response,
        body=None,
    )

    assert policy.get_delay(error) == 30.0


def test_retry_after_invalid_uses_fallback():
    policy = RetryPolicy(
        RetryConfig(backoff_seconds=0.5)
    )

    request = httpx.Request(
        "POST",
        "https://openrouter.ai/api/v1/chat/completions",
    )

    response = httpx.Response(
        429,
        headers={"Retry-After": "invalid"},
        request=request,
    )

    error = APIStatusError(
        "Rate limited",
        response=response,
        body=None,
    )

    assert policy.get_delay(error) == 0.5


def test_retry_delay_without_header():
    policy = RetryPolicy(
        RetryConfig(backoff_seconds=0.5)
    )

    error = make_status_error(503)

    assert policy.get_delay(error) == 0.5
