"""The single revision implementation shared by M3, M4 and M5."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.verification._common import (
    LABELS, draft_payload, example_payload, generate_json, sequence, text,
)
from src.verification.errors import RevisionError
from src.verification.schema_adapter import make_revision

if TYPE_CHECKING:
    from contracts.schemas import AnswerDraft, QAInput, RevisionResult, VerificationFeedback
    from contracts.interfaces import LLMClient


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
    Detailed edit reasons live in the raw LLM response logged by C; the returned
    dataclass keeps exactly the four fields specified by the shared contract.
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
    changed = set()
    if answer != original["short_answer"]:
        changed.add("final_answer")
    if explanation != original["explanation"]:
        changed.add("final_explanation")
    traced = set()
    for edit in sequence(data.get("edits"), "edits", RevisionError):
        if not isinstance(edit, dict):
            raise RevisionError("Each edit must be a JSON object.")
        field = edit.get("field")
        if not isinstance(field, str) or field not in changed or field in traced:
            raise RevisionError("An edit must name one changed field exactly once.")
        targets = sequence(edit.get("feedback_target_ids"), "feedback_target_ids", RevisionError)
        if not targets or any(not isinstance(target, str) or target not in actionable_targets for target in targets):
            raise RevisionError("Every edit must refer to actionable input feedback.")
        text(edit.get("reason"), "edit.reason", RevisionError)
        traced.add(field)
    if traced != changed:
        raise RevisionError("Every changed field requires an edit provenance record.")
    return make_revision(original["draft_id"], criticism["feedback_id"], answer, explanation)
