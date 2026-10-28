"""UsageWrapper: intercepts LLM generate calls and records usage dicts.

Stores usage with latency_seconds (matching contracts.LLMUsage field name).

Also writes each call to a raw_calls.jsonl file (if raw_calls_path is set)
for audit and replay.  The raw call entry includes example_id and method_id
set via set_context(), plus the messages sent and response text received.

drain_usage() delegates to the inner client if it has drain_usage; otherwise
it returns and clears the wrapper's own accumulated log.
"""

from __future__ import annotations

import json
import pathlib


class UsageWrapper:
    """Wraps any LLM client, recording one usage dict per generate() call.

    Args:
        client: Any object with a keyword-only generate() method.
        default_model: Used in error log entries when model arg is empty.
        raw_calls_path: Path to write raw call log (JSONL).  Parent directory
            is created if it does not exist.  Pass None to disable logging.
    """

    def __init__(
        self,
        client: object,
        default_model: str = "",
        raw_calls_path: str | None = None,
    ) -> None:
        self._inner = client
        self._log: list[dict] = []
        self._default_model = default_model
        self.raw_calls_path = raw_calls_path
        self._example_id: str = ""
        self._method_id: str = ""

    def set_context(self, example_id: str, method_id: str) -> None:
        """Set the current example and method for raw call logging."""
        self._example_id = example_id
        self._method_id = method_id

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
                entry = {
                    "stage": getattr(usage, "stage", stage),
                    "model": getattr(usage, "model", model),
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "cost_usd": getattr(usage, "cost_usd", None),
                    "latency_seconds": getattr(usage, "latency_seconds", None),
                    "cache_hit": bool(getattr(usage, "cache_hit", False)),
                }
                self._log.append(entry)

                # Write raw call to audit log if path is set
                if self.raw_calls_path:
                    self._write_raw_call(stage, messages, response, entry)

            return response
        except Exception as exc:
            err_entry = {
                "stage": stage,
                "model": model or self._default_model,
                "status": "error",
                "error_type": type(exc).__name__,
            }
            self._log.append(err_entry)

            # Write error entry to raw call log too
            if self.raw_calls_path:
                self._write_raw_call_error(stage, messages, exc)

            raise

    def _write_raw_call(
        self, stage: str, messages: list[dict], response: object, usage_entry: dict
    ) -> None:
        """Write a raw call entry to raw_calls.jsonl."""
        raw_entry = {
            "example_id": self._example_id,
            "method_id": self._method_id,
            "stage": usage_entry.get("stage", stage),
            "messages": messages,
            "response_text": getattr(response, "text", None),
            "input_tokens": usage_entry.get("input_tokens"),
            "output_tokens": usage_entry.get("output_tokens"),
            "cost_usd": usage_entry.get("cost_usd"),
            "latency_seconds": usage_entry.get("latency_seconds"),
            "cache_hit": usage_entry.get("cache_hit"),
            "model": usage_entry.get("model"),
        }
        self._append_raw(raw_entry)

    def _write_raw_call_error(
        self, stage: str, messages: list[dict], exc: Exception
    ) -> None:
        """Write an error entry to raw_calls.jsonl."""
        raw_entry = {
            "example_id": self._example_id,
            "method_id": self._method_id,
            "stage": stage,
            "messages": messages,
            "response_text": None,
            "status": "error",
            "error_type": type(exc).__name__,
        }
        self._append_raw(raw_entry)

    def _append_raw(self, entry: dict) -> None:
        """Append one entry to the raw_calls.jsonl file."""
        if not self.raw_calls_path:
            return
        p = pathlib.Path(self.raw_calls_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

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
