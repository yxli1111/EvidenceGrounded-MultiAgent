"""ExperimentRecord: one row per (example, method) pair, written as JSONL.

Design notes
------------
* All fields are plain Python types (str/int/float/bool/list/dict/None) so
  dataclasses.asdict() + json.dumps() always round-trips cleanly.
* draft_usage tracks the shared generate_draft call; usage tracks all
  method-specific LLM calls.  Separating them avoids 5× double-counting
  when computing total-run cost; P3 cost analysis decides how to allocate.
* A record with status="error" has None for most output fields but still
  appears in every denominator — failures are never silently dropped.
* status="skipped" is reserved for examples explicitly excluded before
  running (e.g. dev/eval boundary enforcement); it also counts in denominators.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentRecord:
    # Identity
    example_id: str
    method_id: str           # "M1" | "M2" | "M3" | "M4" | "M5"
    status: str              # "ok" | "error" | "skipped"
    timestamp: str           # ISO-8601 UTC

    # Input snapshot (for debugging without re-loading dataset)
    question: str = ""

    # AnswerDraft (shared across all methods for same example)
    draft_id: str | None = None
    short_answer: str | None = None
    explanation: str | None = None

    # Final output after revision (equals draft for M1)
    final_answer: str | None = None
    final_explanation: str | None = None

    # Intermediate data (serialised to plain dicts)
    claims: list[dict] | None = None            # AtomicClaims (M5)
    evidence: list[dict] | None = None          # retrieved evidence:
                                                #   M4 = whole-draft evidence
                                                #   M5 = all-claim evidence, flat+deduped
    evidence_by_claim: dict | None = None       # M5 only: {claim_id: [Evidence dicts]}
    feedback: dict | None = None                # VerificationFeedback (M3/M4/M5)

    # LLM usage (list of per-call dicts with keys:
    #   stage, model, input_tokens, output_tokens, cost_usd, latency_ms, cache_hit)
    draft_usage: list[dict] = field(default_factory=list)  # generate_draft call
    usage: list[dict] = field(default_factory=list)        # method-specific calls

    # Run metadata
    model_settings: dict | None = None
    prompt_versions: dict | None = None

    # Error details (populated when status="error")
    error: str | None = None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def to_jsonl_line(record: ExperimentRecord) -> str:
    return json.dumps(dataclasses.asdict(record), ensure_ascii=False)


def from_jsonl_line(line: str) -> ExperimentRecord:
    data = json.loads(line)
    return ExperimentRecord(**data)


def load_records(path: str) -> list[ExperimentRecord]:
    """Load all records from a JSONL file; silently skip blank lines."""
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(from_jsonl_line(line))
    except FileNotFoundError:
        pass
    return records


def load_completed(path: str) -> set[tuple[str, str]]:
    """Return set of (example_id, method_id) pairs that already have a record."""
    return {(r.example_id, r.method_id) for r in load_records(path)}


def append_record(path: str, record: ExperimentRecord) -> None:
    """Append one record to path as a single JSONL line."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(to_jsonl_line(record) + "\n")


def make_error_record(
    example_id: str,
    method_id: str,
    error_msg: str,
    question: str = "",
    model_settings: dict | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        example_id=example_id,
        method_id=method_id,
        status="error",
        timestamp=_now_iso(),
        question=question,
        error=error_msg,
        model_settings=model_settings,
    )
