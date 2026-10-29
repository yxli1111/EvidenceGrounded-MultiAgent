"""The single revision implementation shared by M3, M4 and M5."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from src.verification._common import (
    LABELS, draft_payload, example_payload, generate_json, sequence, text,
)
from src.verification.errors import RevisionError
from src.verification.schema_adapter import make_revision

if TYPE_CHECKING:
    from contracts.schemas import AnswerDraft, QAInput, RevisionResult, VerificationFeedback
    from contracts.interfaces import LLMClient


_LOG = logging.getLogger(__name__)


def _record_edit_audit(original: dict, criticism: dict, answer: str, explanation: str, edits) -> None:
    """Compute actual changes; model-supplied edit notes are optional metadata.

    Missing or malformed notes never make valid answer fields fail. Do not invent
    a model rationale or attribute an unreported change to a particular target.
    E can capture this logger at INFO for the complete deterministic audit.
    """
    pairs = {
        "final_answer": (original["short_answer"], answer),
        "final_explanation": (original["explanation"], explanation),
    }
    changed = {field for field, (before, after) in pairs.items() if before != after}
    valid_targets = {item["target_id"] for item in criticism["items"]}
    notes = {field: [] for field in changed}
    issues = []
    aliases = {"short_answer": "final_answer", "explanation": "final_explanation"}
    if edits is None:
        edits = []
    if not isinstance(edits, (list, tuple)):
        issues.append("edits is not a list; ignored optional model notes")
        edits = []
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            issues.append(f"edit {index}: not an object")
            continue
        field = edit.get("field")
        if not isinstance(field, str):
            issues.append(f"edit {index}: invalid field")
            continue
        field = aliases.get(field, field)
        if field not in pairs:
            issues.append(f"edit {index}: unknown field")
            continue
        if field not in changed:
            issues.append(f"edit {index}: field did not change")
            continue
        targets = edit.get("feedback_target_ids")
        reason = edit.get("reason")
        if (
            not isinstance(targets, (list, tuple)) or not targets
            or any(not isinstance(target, str) or target not in valid_targets for target in targets)
            or not isinstance(reason, str) or not reason.strip()
        ):
            issues.append(f"edit {index}: invalid optional targets/reason")
            continue
        # Multiple explanations of one changed field are valid, not a duplicate edit error.
        notes[field].append({"feedback_target_ids": list(dict.fromkeys(targets)), "reason": reason})
    changes = []
    for field, (before, after) in pairs.items():
        if field not in changed:
            continue
        if not notes[field]:
            issues.append(f"{field}: no valid model note; change recorded by deterministic diff")
        changes.append({
            "field": field, "before": before, "after": after,
            "source": "computed_diff", "model_notes": notes[field],
        })
    audit = {
        "draft_id": original["draft_id"], "feedback_id": criticism["feedback_id"],
        "changes": changes, "issues": issues,
    }
    if issues:
        _LOG.warning("D revision audit metadata: draft_id=%s feedback_id=%s issues=%s",
                     original["draft_id"], criticism["feedback_id"], issues)
    _LOG.info("D revision audit: %s", json.dumps(audit, ensure_ascii=False))


def _feedback_payload(feedback, draft_id: str, context: dict) -> dict:
    known_citations = {f"{title}::{index}" for title, index in context}
    items = []
    for item in sequence(feedback.items, "feedback.items", RevisionError):
        scope = item.scope
        if scope not in ("draft", "claim"):
            raise RevisionError("Feedback scope must be draft or claim.")
        target_id = text(item.target_id, "feedback.target_id", RevisionError)
        if scope == "draft" and target_id != draft_id:
            raise RevisionError("Draft-scoped feedback targets a different draft.")
        label = item.label
        if label is not None and (not isinstance(label, str) or label not in LABELS):
            raise RevisionError("Invalid feedback label.")
        citations = sequence(item.evidence_ids, "feedback.evidence_ids", RevisionError)
        for citation in citations:
            if not isinstance(citation, str) or citation not in known_citations:
                raise RevisionError("Feedback citation is not in the example's context.")
        if len(set(citations)) != len(citations):
            raise RevisionError("Duplicate feedback citation.")
        if label is None and citations:
            raise RevisionError("Unlabelled M3 criticism must not invent evidence citations.")
        if label in {"SUPPORTED", "CONTRADICTED"} and not citations:
            raise RevisionError("Evidence verdicts require citations.")
        correction = getattr(item, "suggested_correction", None)
        if correction is not None:
            correction = text(correction, "suggested_correction", RevisionError)
        items.append({
            "scope": scope, "target_id": target_id, "label": label,
            "rationale": text(item.rationale, "rationale", RevisionError),
            "evidence_ids": citations, "suggested_correction": correction,
        })
    return {
        "feedback_id": text(feedback.feedback_id, "feedback_id", RevisionError),
        "items": items,
        "overall_comment": text(feedback.overall_comment, "overall_comment", RevisionError, allow_empty=True),
    }


def _actionable(item: dict) -> bool:
    return (
        item["label"] in (None, "CONTRADICTED")
        or (item["label"] == "UNSUPPORTED" and item["suggested_correction"] is not None)
    )


def revise_answer(
    example: QAInput, draft: AnswerDraft, feedback: VerificationFeedback, llm: LLMClient
) -> RevisionResult:
    """At most one generation. E must call this only once per method/example.

    No method argument or method-specific settings exist. Empty/positive feedback,
    or insufficient evidence without a proposed qualification, preserves the draft.
    Actual edits are computed from the before/after text and logged; optional model
    reasons live in that audit and C's raw response. The returned dataclass keeps
    exactly the four fields specified by the shared contract.
    """
    question, context = example_payload(example, RevisionError)
    original = draft_payload(draft, RevisionError)
    criticism = _feedback_payload(feedback, original["draft_id"], context)
    unchanged = make_revision(
        original["draft_id"], criticism["feedback_id"],
        original["short_answer"], original["explanation"],
    )
    actionable_targets = {item["target_id"] for item in criticism["items"] if _actionable(item)}
    if not actionable_targets:
        return unchanged

    data = generate_json(llm, "revise_answer", {
        "example": question, "draft": original, "feedback": criticism,
    }, RevisionError)
    answer = text(data.get("final_answer"), "final_answer", RevisionError, allow_empty=True)
    explanation = text(data.get("final_explanation"), "final_explanation", RevisionError, allow_empty=True)
    _record_edit_audit(original, criticism, answer, explanation, data.get("edits"))
    return make_revision(original["draft_id"], criticism["feedback_id"], answer, explanation)
