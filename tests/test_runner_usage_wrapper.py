"""Tests for UsageWrapper (src/runner/usage_wrapper.py).

Verifies:
- keyword-only forwarding of generate() to inner client
- latency_seconds → latency_ms conversion in recorded dict
- error logging on generate() failure
- drain_usage() delegation to inner when inner has drain_usage
- drain_usage() uses own log when inner has no drain_usage
- own log cleared after drain
"""

import pytest

from src.runner.usage_wrapper import UsageWrapper


# ---------------------------------------------------------------------------
# Helper stubs
# ---------------------------------------------------------------------------

class _FakeUsage:
    def __init__(
        self,
        stage="test_stage",
        model="m1",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        latency_seconds=0.12,
        cache_hit=False,
    ):
        self.stage = stage
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cost_usd = cost_usd
        self.latency_seconds = latency_seconds
        self.cache_hit = cache_hit


class _FakeResponse:
    def __init__(self, usage=None):
        self.text = "response text"
        self.usage = usage


class _InnerClient:
    """Basic inner client: no drain_usage.  Records last call kwargs."""

    def __init__(self, response=None, raise_exc=None):
        self._static_response = response  # if set, always return this
        self._raise = raise_exc
        self.last_kwargs: dict = {}

    def generate(self, *, stage, messages, model, temperature, max_tokens):
        self.last_kwargs = dict(
            stage=stage,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if self._raise is not None:
            raise self._raise
        if self._static_response is not None:
            return self._static_response
        # Default: return a response whose usage.stage mirrors the call
        return _FakeResponse(_FakeUsage(stage=stage, model=model))


class _InnerClientWithDrain(_InnerClient):
    """Inner client that also has drain_usage."""

    def __init__(self, *args, drain_result=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._drain_result = drain_result if drain_result is not None else [
            {"stage": "inner_stage", "cost_usd": 0.002}
        ]

    def drain_usage(self) -> list[dict]:
        return list(self._drain_result)


# ---------------------------------------------------------------------------
# Forwarding
# ---------------------------------------------------------------------------

class TestGenerateForwarding:
    def test_generate_forwards_keyword_args(self):
        inner = _InnerClient()
        wrapper = UsageWrapper(inner)
        wrapper.generate(
            stage="my_stage",
            messages=[{"role": "user"}],
            model="gpt-4o",
            temperature=0.7,
            max_tokens=256,
        )
        assert inner.last_kwargs == {
            "stage": "my_stage",
            "messages": [{"role": "user"}],
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 256,
        }

    def test_generate_returns_inner_response(self):
        resp = _FakeResponse(_FakeUsage())
        inner = _InnerClient(response=resp)
        wrapper = UsageWrapper(inner)
        result = wrapper.generate(
            stage="s", messages=[], model="m", temperature=0.0, max_tokens=10
        )
        assert result is resp


# ---------------------------------------------------------------------------
# Latency conversion
# ---------------------------------------------------------------------------

class TestLatencyConversion:
    def test_success_converts_latency_seconds_to_ms(self):
        usage = _FakeUsage(latency_seconds=0.12)
        inner = _InnerClient(response=_FakeResponse(usage))
        wrapper = UsageWrapper(inner)
        wrapper.generate(stage="s", messages=[], model="m", temperature=0.0, max_tokens=10)
        log = wrapper.drain_usage()
        assert len(log) == 1
        assert log[0]["latency_ms"] == pytest.approx(120.0)

    def test_none_latency_stays_none(self):
        usage = _FakeUsage(latency_seconds=None)
        inner = _InnerClient(response=_FakeResponse(usage))
        wrapper = UsageWrapper(inner)
        wrapper.generate(stage="s", messages=[], model="m", temperature=0.0, max_tokens=10)
        log = wrapper.drain_usage()
        assert log[0]["latency_ms"] is None

    def test_no_usage_object_produces_no_log_entry(self):
        inner = _InnerClient(response=_FakeResponse(usage=None))
        wrapper = UsageWrapper(inner)
        wrapper.generate(stage="s", messages=[], model="m", temperature=0.0, max_tokens=10)
        log = wrapper.drain_usage()
        assert log == []


# ---------------------------------------------------------------------------
# Error logging
# ---------------------------------------------------------------------------

class TestErrorLogging:
    def test_error_entry_logged_with_error_type(self):
        inner = _InnerClient(raise_exc=ValueError("bad input"))
        wrapper = UsageWrapper(inner)
        with pytest.raises(ValueError):
            wrapper.generate(stage="err_stage", messages=[], model="m",
                             temperature=0.0, max_tokens=10)
        log = wrapper.drain_usage()
        assert len(log) == 1
        assert log[0]["error_type"] == "ValueError"
        assert log[0]["stage"] == "err_stage"

    def test_exception_reraised(self):
        inner = _InnerClient(raise_exc=RuntimeError("oops"))
        wrapper = UsageWrapper(inner)
        with pytest.raises(RuntimeError, match="oops"):
            wrapper.generate(stage="s", messages=[], model="m",
                             temperature=0.0, max_tokens=10)

    def test_error_entry_includes_model(self):
        inner = _InnerClient(raise_exc=KeyError("missing"))
        wrapper = UsageWrapper(inner, default_model="fallback-model")
        with pytest.raises(KeyError):
            wrapper.generate(stage="s", messages=[], model="", temperature=0.0, max_tokens=10)
        log = wrapper.drain_usage()
        assert log[0]["model"] == "fallback-model"


# ---------------------------------------------------------------------------
# drain_usage delegation
# ---------------------------------------------------------------------------

class TestDrainUsage:
    def test_drain_uses_inner_drain_when_available(self):
        drain_result = [{"stage": "inner", "cost_usd": 0.002}]
        inner = _InnerClientWithDrain(drain_result=drain_result)
        wrapper = UsageWrapper(inner)
        wrapper.generate(stage="s", messages=[], model="m", temperature=0.0, max_tokens=10)
        result = wrapper.drain_usage()
        assert result == drain_result

    def test_drain_uses_own_log_when_no_inner_drain(self):
        usage = _FakeUsage(stage="own_stage", latency_seconds=0.05)
        inner = _InnerClient(response=_FakeResponse(usage))
        wrapper = UsageWrapper(inner)
        wrapper.generate(stage="own_stage", messages=[], model="m",
                         temperature=0.0, max_tokens=10)
        result = wrapper.drain_usage()
        assert len(result) == 1
        assert result[0]["stage"] == "own_stage"

    def test_own_log_cleared_after_drain(self):
        usage = _FakeUsage()
        inner = _InnerClient(response=_FakeResponse(usage))
        wrapper = UsageWrapper(inner)
        wrapper.generate(stage="s", messages=[], model="m", temperature=0.0, max_tokens=10)
        wrapper.drain_usage()
        second = wrapper.drain_usage()
        assert second == []

    def test_inner_drain_used_discards_wrapper_log(self):
        usage = _FakeUsage()
        inner = _InnerClientWithDrain(
            response=_FakeResponse(usage),
            drain_result=[{"stage": "inner_only"}],
        )
        wrapper = UsageWrapper(inner)
        wrapper.generate(stage="s", messages=[], model="m", temperature=0.0, max_tokens=10)
        result = wrapper.drain_usage()
        # Inner's records returned, not wrapper log
        assert result == [{"stage": "inner_only"}]

    def test_multiple_calls_accumulated_before_drain(self):
        inner = _InnerClient()
        wrapper = UsageWrapper(inner)
        for i in range(3):
            wrapper.generate(stage=f"stage_{i}", messages=[], model="m",
                             temperature=0.0, max_tokens=10)
        log = wrapper.drain_usage()
        assert len(log) == 3
        stages = [e["stage"] for e in log]
        assert stages == ["stage_0", "stage_1", "stage_2"]
