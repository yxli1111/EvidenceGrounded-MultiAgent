import os
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv


class KeyUsageError(Exception):
    """Raised when OpenRouter key usage cannot be queried."""
    pass


@dataclass(frozen=True)
class KeyUsage:
    usage: float | None
    limit: float | None
    limit_remaining: float | None


def get_key_usage(
    api_key: str | None = None,
    *,
    timeout: float = 15.0,
    http_client: httpx.Client | None = None,
) -> KeyUsage:
    """Query OpenRouter API key usage information."""

    load_dotenv()

    key = api_key or os.getenv("OPENROUTER_API_KEY")

    if not key:
        raise KeyUsageError(
            "OPENROUTER_API_KEY is not configured."
        )

    try:
        if http_client is None:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(
                    "https://openrouter.ai/api/v1/key",
                    headers={
                        "Authorization": f"Bearer {key}"
                    },
                )
        else:
            response = http_client.get(
                "https://openrouter.ai/api/v1/key",
                headers={
                    "Authorization": f"Bearer {key}"
                },
                timeout=timeout,
            )

        response.raise_for_status()
        data = response.json()["data"]

        return KeyUsage(
            usage=(
                float(data["usage"])
                if data.get("usage") is not None
                else None
            ),
            limit=(
                float(data["limit"])
                if data.get("limit") is not None
                else None
            ),
            limit_remaining=(
                float(data["limit_remaining"])
                if data.get("limit_remaining") is not None
                else None
            ),
        )

    except (
        httpx.HTTPError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        raise KeyUsageError(
            "Failed to retrieve OpenRouter key usage."
        ) from exc
