import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from contracts.schemas import LLMUsage


class UsageLogger:
    """Save successful and failed LLM request logs."""

    def __init__(
        self,
        log_path: str = "logs/llm_usage.jsonl",
        failure_log_path: str = "logs/llm_failures.jsonl",
    ):
        self.log_path = Path(log_path)
        self.failure_log_path = Path(failure_log_path)

        # Create log directories if necessary.
        self.log_path.parent.mkdir(
            parents=True, exist_ok=True
        )
        self.failure_log_path.parent.mkdir(
            parents=True, exist_ok=True
        )

    def _append_jsonl(self, path: Path, record: dict) -> None:
        """Append one JSON record to a JSONL file."""
        with path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False,
                ) + "\n"
            )

    def log(self, usage: LLMUsage) -> None:
        """Record a successful LLM request."""
        record = asdict(usage)
        record["timestamp"] = datetime.now(
            timezone.utc
        ).isoformat()

        self._append_jsonl(self.log_path, record)

    def log_failure(
        self,
        *,
        stage: str,
        model: str,
        error_type: str,
        estimated_cost_usd: float | None,
        actual_cost_usd: float | None,
        latency_seconds: float,
    ) -> None:
        """Record a failed LLM request without sensitive content."""

        record = {
            "stage": stage,
            "model": model,
            "error_type": error_type,
            "estimated_cost_usd": estimated_cost_usd,
            "actual_cost_usd": actual_cost_usd,
            "latency_seconds": latency_seconds,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        self._append_jsonl(
            self.failure_log_path,
            record,
        )
