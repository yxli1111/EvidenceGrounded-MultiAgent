import os

import httpx
from dotenv import load_dotenv


def format_usd(value):
    """Format a USD amount, preserving small values."""
    if value is None:
        return "Not available"
    return f"${float(value):.6f}"


def main():
    # Load API key from the project .env file
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is missing.")
        return

    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/key",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            timeout=15.0,
        )

        response.raise_for_status()
        data = response.json()["data"]

    except (httpx.HTTPError, ValueError, KeyError) as exc:
        print(
            "Failed to query OpenRouter key usage:",
            type(exc).__name__,
        )
        return

    usage = data.get("usage")
    limit = data.get("limit")
    remaining = data.get("limit_remaining")

    print("\n" + "=" * 45)
    print("       OpenRouter API Key Usage")
    print("=" * 45)

    print("Usage (USD):          ", format_usd(usage))
    print("Key Limit (USD):      ", format_usd(limit))
    print("Key Remaining (USD):  ", format_usd(remaining))

    if limit is None:
        print("\nNote: No per-key spending limit is reported.")
        print("This does not indicate the account balance.")

    print("=" * 45)


if __name__ == "__main__":
    main()
