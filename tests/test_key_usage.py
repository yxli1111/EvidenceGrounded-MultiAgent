from unittest.mock import MagicMock

import pytest
import httpx

from src.core.key_usage import (
    KeyUsageError,
    get_key_usage,
)


def make_mock_http_client(data):
    client = MagicMock()

    response = MagicMock()
    response.json.return_value = {"data": data}
    response.raise_for_status.return_value = None

    client.get.return_value = response
    return client


def test_get_key_usage_success():
    client = make_mock_http_client({
        "usage": 0.000005,
        "limit": 15.0,
        "limit_remaining": 14.999995,
    })

    result = get_key_usage(
        api_key="mock-test-key",
        http_client=client,
    )

    assert result.usage == pytest.approx(0.000005)
    assert result.limit == pytest.approx(15.0)
    assert result.limit_remaining == pytest.approx(
        14.999995
    )

    client.get.assert_called_once()


def test_get_key_usage_without_limit():
    client = make_mock_http_client({
        "usage": 0.5,
        "limit": None,
        "limit_remaining": None,
    })

    result = get_key_usage(
        api_key="mock-test-key",
        http_client=client,
    )

    assert result.usage == 0.5
    assert result.limit is None
    assert result.limit_remaining is None


def test_get_key_usage_http_failure():
    client = MagicMock()

    client.get.side_effect = httpx.ConnectError(
        "Mock network failure"
    )

    with pytest.raises(KeyUsageError):
        get_key_usage(
            api_key="mock-test-key",
            http_client=client,
        )


def test_get_key_usage_invalid_response():
    client = make_mock_http_client({
        "usage": "invalid-number",
        "limit": 15.0,
        "limit_remaining": 14.0,
    })

    with pytest.raises(KeyUsageError):
        get_key_usage(
            api_key="mock-test-key",
            http_client=client,
        )
