"""UsageWrapper: intercepts LLM generate calls and records usage dicts.

Converts contracts.LLMUsage.latency_seconds (float) → latency_ms (float)
in the recorded dict, bridging C's LLMClient interface to E's cost metrics.

drain_usage() delegates to the inner client if it has drain_usage; otherwise
it returns and clears the wrapper's own accumulated log.
"""

from __future__ import annotations


class UsageWrapper:
    """Wraps any LLM client, recording one usage dict per generate() call.

    Args:
        client: Any object with a keyword-only generate() method.
        default_model: Used in error log entries when model arg is empty.
    """

    def __init__(self, client: object, default_model: str = "") -> None:
        self._inner = client
        self._log: list[dict] = []
        self._default_model = default_model

    def generate(
        self,
        *,
        stage: str,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> object:
        """Forward to inner client and record usage.  Exceptions are re-raised
        after logging an error entry."""
        try:
            response = self._inner.generate(
                stage=stage,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = getattr(response, "usage", None)
            if usage is not None:
                lat_s = getattr(usage, "latency_seconds", None)
                lat_ms = lat_s * 1000.0 if lat_s is not None else None
                self._log.append({
                    "stage": getattr(usage, "stage", stage),
                    "model": getattr(usage, "model", model),
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "cost_usd": getattr(usage, "cost_usd", None),
                    "latency_ms": lat_ms,
                    "cache_hit": bool(getattr(usage, "cache_hit", False)),
                })
            return response
        except Exception as exc:
            self._log.append({
                "stage": stage,
                "model": model or self._default_model,
                "status": "error",
                "error_type": type(exc).__name__,
            })
            raise

    def drain_usage(self) -> list[dict]:
        """Return recorded usage and clear the log.

        If the inner client has drain_usage, delegates to it and discards
        the wrapper's own log (the inner client is the authoritative source).
        """
        inner_drain = getattr(self._inner, "drain_usage", None)
        if callable(inner_drain):
            self._log.clear()
            return inner_drain()
        result = list(self._log)
        self._log.clear()
        return result
