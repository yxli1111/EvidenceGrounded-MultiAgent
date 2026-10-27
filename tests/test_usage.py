import json

from contracts.schemas import LLMUsage
from src.core.usage import UsageLogger


def test_usage_logger(tmp_path):
    log_file = tmp_path / "usage.jsonl"

    logger = UsageLogger(str(log_file))

    usage = LLMUsage(
        stage="claim_extraction",
        model="test-model",
        input_tokens=100,
        output_tokens=20,
        cost_usd=0.001,
        latency_seconds=1.5,
        cache_hit=False,
    )

    logger.log(usage)

    assert log_file.exists()

    lines = log_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["stage"] == "claim_extraction"
    assert record["model"] == "test-model"
    assert record["input_tokens"] == 100
    assert record["output_tokens"] == 20
    assert record["cost_usd"] == 0.001
    assert record["cache_hit"] is False
    assert "timestamp" in record


def test_failure_logger(tmp_path):
    success_file = tmp_path / "usage.jsonl"
    failure_file = tmp_path / "failures.jsonl"

    logger = UsageLogger(
        log_path=str(success_file),
        failure_log_path=str(failure_file),
    )

    # Record a simulated failed API request.
    logger.log_failure(
        stage="verification",
        model="openai/gpt-4o-mini",
        error_type="LLMClientError",
        estimated_cost_usd=0.005,
        actual_cost_usd=None,
        latency_seconds=2.5,
    )

    assert failure_file.exists()

    records = failure_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(records) == 1

    record = json.loads(records[0])

    assert record["stage"] == "verification"
    assert record["error_type"] == "LLMClientError"
    assert record["estimated_cost_usd"] == 0.005
    assert record["actual_cost_usd"] is None
    assert record["latency_seconds"] == 2.5
    assert "timestamp" in record

    # A failed request must not appear in success logs.
    assert not success_file.exists()
