"""Persistent run journal for claim distribution experiments.

This module does not call any external API.

It records draft execution states and conservative budget
reservations across process restarts.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


class JournalError(RuntimeError):
    """Raised when the journal is unsafe or inconsistent."""


class JournalBudgetExceeded(JournalError):
    """Raised when the cumulative journal budget is exhausted."""


class JournalUncertainState(JournalError):
    """Raised when an interrupted request needs manual review."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, data: dict) -> None:
    """Replace a JSON file using a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")

    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary, path)

    finally:
        if temporary.exists():
            temporary.unlink()


class RunJournal:
    """Track draft states and cumulative budget reservations.

    This is intended for a single local process at a time.
    It is not a multi-process locking system.
    """

    VERSION = 1

    def __init__(
        self,
        path: str | Path,
        fingerprint: str,
        max_calls: int,
        max_cost_usd: float,
    ):
        self.path = Path(path)
        self.fingerprint = fingerprint

        if max_calls < 1:
            raise ValueError("max_calls must be positive")

        if (
            not math.isfinite(max_cost_usd)
            or max_cost_usd <= 0
        ):
            raise ValueError(
                "max_cost_usd must be positive and finite"
            )

        self.max_calls = max_calls
        self.max_cost_usd = max_cost_usd
        self.data: dict = {}

    def create(self) -> None:
        """Create a new journal; never overwrite an existing one."""
        if self.path.exists():
            raise JournalError(
                "Journal already exists. Resume explicitly."
            )

        self.data = {
            "version": self.VERSION,
            "fingerprint": self.fingerprint,
            "max_calls": self.max_calls,
            "max_cost_usd": self.max_cost_usd,
            "calls": 0,
            "spent_usd": 0.0,
            "uncertain_usd": 0.0,
            "drafts": {},
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }

        self._save()

    def load(self) -> None:
        """Load and validate an existing journal."""
        if not self.path.is_file():
            raise JournalError("Journal file is missing")

        try:
            data = json.loads(
                self.path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise JournalError(
                "Journal cannot be read"
            ) from exc

        if not isinstance(data, dict):
            raise JournalError("Invalid journal structure")

        if data.get("version") != self.VERSION:
            raise JournalError("Unsupported journal version")

        if data.get("fingerprint") != self.fingerprint:
            raise JournalError(
                "Experiment fingerprint changed"
            )

        if data.get("max_calls") != self.max_calls:
            raise JournalError(
                "Cumulative call limit changed"
            )

        if data.get("max_cost_usd") != self.max_cost_usd:
            raise JournalError(
                "Cumulative cost limit changed"
            )

        drafts = data.get("drafts")

        if not isinstance(drafts, dict):
            raise JournalError("Invalid draft states")

        calls = data.get("calls")
        spent = data.get("spent_usd")
        uncertain = data.get("uncertain_usd")

        if (
            type(calls) is not int
            or calls < 0
            or calls > self.max_calls
        ):
            raise JournalError("Invalid call count")

        for value in (spent, uncertain):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise JournalError("Invalid cost record")

        if spent + uncertain > self.max_cost_usd + 1e-9:
            raise JournalError(
                "Recorded cost exceeds journal budget"
            )

        allowed = {"in_progress", "success", "error"}

        for draft_id, record in drafts.items():
            if (
                not isinstance(draft_id, str)
                or not isinstance(record, dict)
                or record.get("status") not in allowed
            ):
                raise JournalError("Invalid draft record")

        self.data = data

    def _save(self) -> None:
        self.data["updated_at"] = utc_now()
        atomic_write_json(self.path, self.data)

    @property
    def remaining_calls(self) -> int:
        return self.max_calls - self.data["calls"]

    @property
    def remaining_cost_usd(self) -> float:
        return max(
            0.0,
            self.max_cost_usd
            - self.data["spent_usd"]
            - self.data["uncertain_usd"],
        )

    def assert_safe_to_resume(self) -> None:
        """Never automatically retry interrupted requests."""
        for draft_id, record in self.data["drafts"].items():
            if record["status"] == "in_progress":
                raise JournalUncertainState(
                    f"Draft {draft_id} is in_progress. "
                    "Check API usage before resuming."
                )

    def start(
        self,
        draft_id: str,
        reserved_calls: int,
        reserved_usd: float,
    ) -> None:
        """Reserve a conservative envelope before a draft."""
        if not draft_id:
            raise ValueError("draft_id cannot be empty")

        if reserved_calls < 1:
            raise ValueError("reserved_calls must be positive")

        if (
            not math.isfinite(reserved_usd)
            or reserved_usd <= 0
        ):
            raise ValueError("Invalid reserved cost")

        if draft_id in self.data["drafts"]:
            raise JournalError(
                "Draft already recorded; manual review required"
            )

        self.assert_safe_to_resume()

        if reserved_calls > self.remaining_calls:
            raise JournalBudgetExceeded(
                "Cumulative call limit reached"
            )

        if reserved_usd > self.remaining_cost_usd + 1e-12:
            raise JournalBudgetExceeded(
                "Cumulative cost limit reached"
            )

        # Persist the reservation before the caller sends requests.
        self.data["calls"] += reserved_calls
        self.data["uncertain_usd"] += reserved_usd

        self.data["drafts"][draft_id] = {
            "status": "in_progress",
            "reserved_calls": reserved_calls,
            "reserved_usd": reserved_usd,
            "started_at": utc_now(),
        }

        self._save()

    def finish(
        self,
        draft_id: str,
        status: str,
        claim_count: int | None,
        actual_calls: int,
        actual_cost_usd: float | None,
        error_type: str = "",
    ) -> None:
        """Settle a draft only when its usage is known."""
        record = self.data["drafts"].get(draft_id)

        if not record or record["status"] != "in_progress":
            raise JournalError(
                "Draft is not in_progress"
            )

        if status not in {"success", "error"}:
            raise ValueError("Invalid final status")

        if status == "success":
            if (
                type(claim_count) is not int
                or claim_count < 0
            ):
                raise ValueError("Invalid successful claim count")
        elif claim_count is not None:
            raise ValueError(
                "Failed draft must not have a claim count"
            )

        if (
            type(actual_calls) is not int
            or actual_calls < 0
            or actual_calls > record["reserved_calls"]
        ):
            raise ValueError("Invalid actual call count")

        if (
            actual_cost_usd is None
            or not math.isfinite(actual_cost_usd)
            or actual_cost_usd < 0
        ):
            raise JournalUncertainState(
                "Actual cost is unknown; keep reservation"
            )

        new_spent = (
            self.data["spent_usd"] + actual_cost_usd
        )

        new_uncertain = (
            self.data["uncertain_usd"]
            - record["reserved_usd"]
        )

        if (
            new_spent + new_uncertain
            > self.max_cost_usd + 1e-9
        ):
            raise JournalUncertainState(
                "Actual cost exceeds reserved budget"
            )

        self.data["spent_usd"] = new_spent
        self.data["uncertain_usd"] = max(0.0, new_uncertain)

        # Refund unused call slots.
        self.data["calls"] -= (
            record["reserved_calls"] - actual_calls
        )

        record.update({
            "status": status,
            "claim_count": claim_count,
            "actual_calls": actual_calls,
            "actual_cost_usd": actual_cost_usd,
            "error_type": error_type,
            "finished_at": utc_now(),
        })

        self._save()

    def get_status(self, draft_id: str) -> str | None:
        record = self.data["drafts"].get(draft_id)
        return None if record is None else record["status"]

    def summary(self) -> dict:
        return {
            "calls": self.data["calls"],
            "spent_usd": self.data["spent_usd"],
            "uncertain_usd": self.data["uncertain_usd"],
            "remaining_calls": self.remaining_calls,
            "remaining_cost_usd": self.remaining_cost_usd,
        }