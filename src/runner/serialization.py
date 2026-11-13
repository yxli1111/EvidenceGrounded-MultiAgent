"""Public serialization helpers for ExperimentRecord and port objects.

These were previously private helpers in runner.py (_ser_draft, etc.).
Moving them here makes them importable by other modules (A/B/C/D integration
tests, replay tools) without importing the full runner.

Usage::

    from src.runner.serialization import ser_draft, ser_revision, to_jsonable
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

from .records import ExperimentRecord


def ser_draft(d: Any) -> dict:
    """Serialize an AnswerDraft object (or dict) to a plain dict."""
    if isinstance(d, dict):
        return d
    return {
        "draft_id": d.draft_id,
        "short_answer": d.short_answer,
        "explanation": d.explanation,
    }


def ser_revision(r: Any) -> dict:
    """Serialize a RevisionResult object (or dict) to a plain dict."""
    if isinstance(r, dict):
        return r
    return {
        "draft_id": r.draft_id,
        "feedback_id": r.feedback_id,
        "final_answer": r.final_answer,
        "final_explanation": r.final_explanation,
    }


def ser_feedback(fb: Any) -> dict:
    """Serialize a VerificationFeedback object (or dict) to a plain dict."""
    if isinstance(fb, dict):
        return fb
    items = []
    for item in fb.items:
        items.append({
            "scope": item.scope,
            "target_id": item.target_id,
            "rationale": item.rationale,
            "label": item.label,
            "evidence_ids": list(item.evidence_ids),
            "suggested_correction": getattr(item, "suggested_correction", None),
        })
    return {
        "feedback_id": fb.feedback_id,
        "items": items,
        "overall_comment": fb.overall_comment,
    }


def ser_evidence_list(evs: list) -> list[dict]:
    """Serialize a list of Evidence objects (or dicts) to a list of plain dicts."""
    result = []
    for ev in evs:
        if isinstance(ev, dict):
            result.append(ev)
        else:
            result.append({
                "doc_title": ev.doc_title,
                "sentence_id": ev.sentence_id,
                "text": ev.text,
                "score": float(ev.score),
                "retrieval_query": ev.retrieval_query,
                "rank": ev.rank,
            })
    return result


def ser_claims(claims: list) -> list[dict]:
    """Serialize a list of AtomicClaim objects (or dicts) to a list of plain dicts."""
    result = []
    for c in claims:
        if isinstance(c, dict):
            result.append(c)
        else:
            result.append({
                "claim_id": c.claim_id,
                "claim_text": c.claim_text,
                "source_span": list(c.source_span) if getattr(c, "source_span", None) else None,
            })
    return result


def to_jsonable(record: ExperimentRecord) -> dict:
    """Convert an ExperimentRecord to a JSON-serializable dict."""
    return dataclasses.asdict(record)


def record_to_jsonl_line(record: ExperimentRecord) -> str:
    """Serialize an ExperimentRecord to a JSONL line string."""
    return json.dumps(to_jsonable(record), ensure_ascii=False)
