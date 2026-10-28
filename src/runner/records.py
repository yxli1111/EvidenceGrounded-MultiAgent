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
import pathlib
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

    # Optional fields added for contracts alignment
    question_type: str | None = None        # "factoid" | "yesno" | etc.
    raw_outputs: list[str] | None = None    # raw LLM responses on AgentOutputError

    # Retry tracking: attempt 1 = first run; incremented on each retry
    attempt: int = 1


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def to_jsonl_line(record: ExperimentRecord) -> str:
    return json.dumps(dataclasses.asdict(record), ensure_ascii=False)


def from_jsonl_line(line: str) -> ExperimentRecord:
    data = json.loads(line)
    data.setdefault("attempt", 1)  # backward compat: old records have no attempt field
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


def load_completed(path: str, retry_errors: bool = False) -> set[tuple[str, str]]:
    """Return set of (example_id, method_id) pairs that are considered done.

    When retry_errors=False (default): any existing record = done.
    When retry_errors=True: only pairs whose *latest* attempt has status="ok"
    are considered done; error records are re-run.
    """
    records = load_records(path)
    if not retry_errors:
        return {(r.example_id, r.method_id) for r in records}
    latest: dict[tuple, ExperimentRecord] = {}
    for r in records:
        key = (r.example_id, r.method_id)
        if key not in latest or r.attempt > latest[key].attempt:
            latest[key] = r
    return {key for key, rec in latest.items() if rec.status == "ok"}


def latest_records(records: list[ExperimentRecord]) -> list[ExperimentRecord]:
    """Return the highest-attempt record per (example_id, method_id) pair.

    Use this to compute accuracy metrics (EM, F1, Recall@k) so that retried
    examples are counted once.  Cost metrics should use all records so that
    retry waste is not hidden.
    """
    latest: dict[tuple, ExperimentRecord] = {}
    for r in records:
        key = (r.example_id, r.method_id)
        if key not in latest or r.attempt > latest[key].attempt:
            latest[key] = r
    return list(latest.values())


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


# ---------------------------------------------------------------------------
# Lightweight draft dataclass (fallback when contracts.schemas is unavailable)
# ---------------------------------------------------------------------------

@dataclass
class _DraftFallback:
    """Substitute for contracts.schemas.AnswerDraft when that package is absent."""
    draft_id: str
    short_answer: str
    explanation: str


# ---------------------------------------------------------------------------
# Draft cache helpers
# ---------------------------------------------------------------------------

def drafts_path(output_path: str) -> str:
    """Return the path to the drafts cache file alongside output_path."""
    return str(pathlib.Path(output_path).parent / "drafts.jsonl")


def load_draft_cache(path: str) -> dict[str, dict]:
    """Load draft cache from a JSONL file; returns empty dict if file missing."""
    cache: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    cache[d["example_id"]] = d
    except FileNotFoundError:
        pass
    return cache


def save_draft(
    path: str,
    example_id: str,
    draft_dict: dict,
    draft_usage: list[dict],
) -> None:
    """Append one draft entry to the drafts cache JSONL file."""
    record = {
        "example_id": example_id,
        "draft_id": draft_dict["draft_id"],
        "short_answer": draft_dict["short_answer"],
        "explanation": draft_dict["explanation"],
        "draft_usage": draft_usage,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
