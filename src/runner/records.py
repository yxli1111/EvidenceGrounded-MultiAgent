"""ExperimentRecord: one row per (example, method) pair, written as JSONL.

Design notes
------------
* All fields are plain Python types (str/int/float/bool/list/dict/None) so
  dataclasses.asdict() + json.dumps() always round-trips cleanly.
* draft_usage tracks the shared generate_draft call; usage tracks all
  method-specific LLM calls.  Separating them avoids 5× double-counting
  when computing total-run cost; P3 cost analysis decides how to allocate.
* A record with status="error" has None for most output fields but still
  appears in every metric denominator — failures are never silently dropped.
* status="skipped" is reserved for examples explicitly excluded before
  running (e.g. dev/eval boundary enforcement); it also counts in denominators.

Field naming conventions (Phase 2 alignment)
---------------------------------------------
* draft: dict | None — AnswerDraft snapshot {"draft_id", "short_answer", "explanation"}
* final: dict | None — RevisionResult snapshot {"draft_id", "feedback_id",
  "final_answer", "final_explanation"}.  M1 MUST be None (no RevisionResult).
* evidence: dict — keyed by target ID.  M4: {draft_id: [Evidence dicts]}.
  M5: {claim_id: [Evidence dicts]}.  Empty dict when no evidence.
* model_config: renamed from model_settings (Phase 2 alignment with contracts).
* raw_calls_path: relative path to raw_calls.jsonl alongside the run output.
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

    # AnswerDraft snapshot: {"draft_id": str, "short_answer": str, "explanation": str}
    draft: dict | None = None

    # RevisionResult snapshot: {"draft_id", "feedback_id", "final_answer", "final_explanation"}
    # M1 MUST be None (no RevisionResult in contract)
    final: dict | None = None

    # Intermediate data
    claims: list[dict] | None = None            # AtomicClaims (M5)

    # Evidence keyed by target: M4 = {draft_id: [...]}, M5 = {claim_id: [...]}
    evidence: dict = field(default_factory=dict)

    feedback: dict | None = None                # VerificationFeedback (M3/M4/M5)

    # LLM usage (list of per-call dicts with keys:
    #   stage, model, input_tokens, output_tokens, cost_usd, latency_seconds, cache_hit)
    draft_usage: list[dict] = field(default_factory=list)  # generate_draft call
    usage: list[dict] = field(default_factory=list)        # method-specific calls

    # Run metadata
    model_config: dict | None = None
    prompt_versions: dict | None = None

    # Error details (populated when status="error")
    error: str | None = None

    # Optional fields added for contracts alignment
    question_type: str | None = None        # "factoid" | "yesno" | etc.
    raw_outputs: list[str] | None = None    # raw LLM responses on AgentOutputError

    # Retry tracking: attempt 1 = first run; incremented on each retry
    attempt: int = 1

    # Reference to raw_calls.jsonl (relative path)
    raw_calls_path: str | None = None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def to_jsonl_line(record: ExperimentRecord) -> str:
    return json.dumps(dataclasses.asdict(record), ensure_ascii=False)


def from_jsonl_line(line: str) -> ExperimentRecord:
    data = json.loads(line)
    data.setdefault("attempt", 1)
    data.setdefault("raw_calls_path", None)

    # Migrate flat draft fields → draft dict
    if "draft" not in data:
        did = data.pop("draft_id", None)
        sa  = data.pop("short_answer", None)
        ex  = data.pop("explanation", None)
        data["draft"] = {"draft_id": did, "short_answer": sa, "explanation": ex} if did else None
    else:
        data.pop("draft_id", None)
        data.pop("short_answer", None)
        data.pop("explanation", None)

    # Migrate flat final fields → final dict
    if "final" not in data:
        fa = data.pop("final_answer", None)
        fe = data.pop("final_explanation", None)
        if data.get("method_id") == "M1":
            data["final"] = None
        elif fa is not None or fe is not None:
            draft_id_val = (data.get("draft") or {}).get("draft_id", "")
            data["final"] = {"draft_id": draft_id_val, "feedback_id": "", "final_answer": fa, "final_explanation": fe}
        else:
            data["final"] = None
    else:
        data.pop("final_answer", None)
        data.pop("final_explanation", None)

    # Migrate evidence list + evidence_by_claim → evidence dict
    if "evidence" in data and isinstance(data["evidence"], list):
        ev_list = data.pop("evidence") or []
        ev_by_claim = data.pop("evidence_by_claim", None)
        mid = data.get("method_id", "")
        if mid == "M5" and ev_by_claim:
            data["evidence"] = ev_by_claim
        elif mid == "M4":
            draft_id_val = (data.get("draft") or {}).get("draft_id", "_m4")
            data["evidence"] = {draft_id_val: ev_list} if ev_list else {}
        else:
            data["evidence"] = {"_legacy": ev_list} if ev_list else {}
    elif "evidence" not in data:
        ev_by_claim = data.pop("evidence_by_claim", None)
        data["evidence"] = ev_by_claim if ev_by_claim else {}
    else:
        data.pop("evidence_by_claim", None)

    # Migrate model_settings → model_config
    if "model_config" not in data and "model_settings" in data:
        data["model_config"] = data.pop("model_settings")
    else:
        data.pop("model_settings", None)

    # Filter to only known fields (prevents TypeError on unrecognised keys from old records)
    valid = {f.name for f in dataclasses.fields(ExperimentRecord)}
    data = {k: v for k, v in data.items() if k in valid}

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
    model_config: dict | None = None,
) -> ExperimentRecord:
    return ExperimentRecord(
        example_id=example_id,
        method_id=method_id,
        status="error",
        timestamp=_now_iso(),
        question=question,
        error=error_msg,
        model_config=model_config,
    )


# ---------------------------------------------------------------------------
# Conversion to contracts.schemas.ExperimentRecord
# ---------------------------------------------------------------------------

def to_contract_record(rec: "ExperimentRecord"):
    """Convert E's ExperimentRecord to contracts.schemas.ExperimentRecord.

    Requires contracts.schemas to be importable (from origin/feat/contracts-v0.1).
    Raises ImportError if the contracts package is not on sys.path.

    M1: final=None (no RevisionResult).
    Evidence dict values are converted from plain dicts to Evidence objects.
    Usage list is converted from plain dicts to LLMUsage objects.
    """
    try:
        from contracts import schemas
    except ImportError as exc:
        raise ImportError(
            "contracts.schemas is required for to_contract_record(); "
            "merge origin/feat/contracts-v0.1 or add contracts/ to sys.path."
        ) from exc

    # draft
    draft = None
    if rec.draft:
        draft = schemas.AnswerDraft(
            draft_id=rec.draft.get("draft_id", ""),
            short_answer=rec.draft.get("short_answer", ""),
            explanation=rec.draft.get("explanation", ""),
        )

    # final (M1 keeps None)
    final = None
    if rec.final:
        final = schemas.RevisionResult(
            draft_id=rec.final.get("draft_id", ""),
            feedback_id=rec.final.get("feedback_id", ""),
            final_answer=rec.final.get("final_answer", ""),
            final_explanation=rec.final.get("final_explanation", ""),
        )

    # claims
    claims = []
    for c in (rec.claims or []):
        if isinstance(c, dict):
            claims.append(schemas.AtomicClaim(
                claim_id=c.get("claim_id", ""),
                claim_text=c.get("claim_text", ""),
                source_span=c.get("source_span"),
            ))
        else:
            claims.append(c)

    # evidence: dict[str, list[Evidence]]
    evidence: dict = {}
    for key, ev_list in (rec.evidence or {}).items():
        converted = []
        for e in (ev_list or []):
            if isinstance(e, dict):
                converted.append(schemas.Evidence(
                    doc_title=e.get("doc_title", ""),
                    sentence_id=e.get("sentence_id", 0),
                    text=e.get("text", ""),
                    score=e.get("score", 0.0),
                    retrieval_query=e.get("retrieval_query", ""),
                    rank=e.get("rank", 0),
                ))
            else:
                converted.append(e)
        evidence[key] = converted

    # usage
    usage = []
    for u in (rec.usage or []):
        if isinstance(u, dict):
            usage.append(schemas.LLMUsage(
                stage=u.get("stage", ""),
                model=u.get("model", ""),
                input_tokens=u.get("input_tokens"),
                output_tokens=u.get("output_tokens"),
                cost_usd=u.get("cost_usd"),
                latency_seconds=u.get("latency_seconds"),
                cache_hit=bool(u.get("cache_hit", False)),
            ))
        else:
            usage.append(u)

    return schemas.ExperimentRecord(
        example_id=rec.example_id,
        method_id=rec.method_id,
        status=rec.status,
        draft=draft,
        final=final,
        claims=claims,
        evidence=evidence,
        feedback=None,  # E stores feedback as plain dict; full conversion is Phase 3
        usage=usage,
        model_config=rec.model_config or {},
        prompt_versions=rec.prompt_versions or {},
        error=rec.error,
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
