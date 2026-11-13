"""Opt-in Phase 2 sufficiency repair with conservative post-verification.

Never changes the V2 revise_answer path. Use only in a separate V3 experiment.
This is a protocol-level safety gate, NOT a factual correctness guarantee.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any

from src.verification._common import draft_payload, example_payload, evidence_payload, generate_json, text
from src.verification.dual_dimension import DualDimensionResult, verify_answer_sufficiency
from src.verification.verifier import verify_draft
from src.verification.errors import RevisionError
from src.verification.schema_adapter import make_revision

REVISION_PROMPT_VERSION = "d-sufficiency-revision-v3"


@dataclass(frozen=True)
class V3RevisionDecision:
    revision: Any
    attempted: bool
    accepted: bool
    reason: str
    candidate_citations: list[str]
    post_sufficiency: str | None = None
    post_support: str | None = None

    def to_dict(self) -> dict:
        r = self.revision
        return {"revision": {"draft_id": r.draft_id, "feedback_id": r.feedback_id,
                              "final_answer": r.final_answer, "final_explanation": r.final_explanation},
                "attempted": self.attempted, "accepted": self.accepted,
                "reason": self.reason, "candidate_citations": list(self.candidate_citations),
                "post_sufficiency": self.post_sufficiency, "post_support": self.post_support,
                "prompt_version": REVISION_PROMPT_VERSION}


def revise_with_sufficiency_v3(example: Any, draft: Any, diagnostic: DualDimensionResult,
                               evidence: list[Any], llm: Any) -> V3RevisionDecision:
    """Attempt one evidence-grounded repair only for INCOMPLETE answers.

    Uses a new LLM call to propose a candidate, then independently invokes
    V3 sufficiency and V2 whole-draft support checks on the candidate. Any
    non-COMPLETE or non-SUPPORTED verdict rolls back to the original answer.
    Errors from transport/invalid model output propagate (not silent successes).
    """
    public, context = example_payload(example, RevisionError)
    original = draft_payload(draft, RevisionError)
    suff = diagnostic.sufficiency
    if suff.question_id != public["example_id"] or suff.draft_id != original["draft_id"]:
        raise RevisionError("Diagnostic belongs to a different question/draft.")
    fb = diagnostic.claim_feedback
    if not isinstance(fb, dict) or not isinstance(fb.get("feedback_id"), str) or not fb["feedback_id"].strip():
        raise RevisionError("V3 diagnostic must contain V2 feedback_id.")
    feedback_id = fb["feedback_id"]
    unchanged = make_revision(original["draft_id"], feedback_id,
                              original["short_answer"], original["explanation"])

    def no_change(reason: str, attempted: bool = False, citations: list[str] | None = None,
                  post_suff: str | None = None, post_support: str | None = None):
        return V3RevisionDecision(unchanged, attempted, False, reason, citations or [], post_suff, post_support)

    if suff.answer_sufficiency != "INCOMPLETE":
        return no_change("not_incomplete")
    if not suff.missing_information or not suff.missing_information.strip():
        raise RevisionError("INCOMPLETE requires missing_information.")
    entries = evidence_payload(evidence, context, RevisionError)
    if not entries or not suff.evidence_ids:
        return no_change("insufficient_evidence_for_repair")
    allowed = {entry["evidence_id"] for entry in entries}
    if not set(suff.evidence_ids).issubset(allowed):
        raise RevisionError("Diagnostic cites evidence not supplied to revision.")
    for i, entry in enumerate(entries, 1):
        entry["citation_id"] = f"R1E{i}"
    alias_to_id = {entry["citation_id"]: entry["evidence_id"] for entry in entries}
    alias_to_id.update({entry["evidence_id"]: entry["evidence_id"] for entry in entries})
    response = generate_json(llm, "revise_sufficiency_v3", {
        "question": public["question"], "draft": original,
        "missing_information": suff.missing_information,
        "evidence": entries, "allowed_evidence_ids": [e["citation_id"] for e in entries],
    }, RevisionError)
    if set(response) != {"final_answer", "final_explanation", "evidence_ids"}:
        raise RevisionError("V3 revision requires exactly final_answer, final_explanation, evidence_ids.")
    answer = text(response["final_answer"], "final_answer", RevisionError)
    explanation = text(response["final_explanation"], "final_explanation", RevisionError)
    citations = response["evidence_ids"]
    if not isinstance(citations, list) or not citations:
        raise RevisionError("V3 candidate must cite at least one supplied evidence sentence.")
    resolved = []
    for citation in citations:
        if not isinstance(citation, str) or citation not in alias_to_id:
            raise RevisionError("V3 candidate cited evidence outside the supplied set.")
        canonical = alias_to_id[citation]
        if canonical not in resolved:
            resolved.append(canonical)
    if answer == original["short_answer"] and explanation == original["explanation"]:
        return no_change("candidate_unchanged", True, resolved)

    # Post-check is deliberately on the proposed answer, never the original.
    from contracts.schemas import AnswerDraft
    candidate = AnswerDraft(original["draft_id"], answer, explanation)
    post_suff = verify_answer_sufficiency(example, candidate, evidence, llm)
    if post_suff.answer_sufficiency != "COMPLETE":
        return no_change("post_sufficiency_not_complete", True, resolved, post_suff.answer_sufficiency)
    post_support = verify_draft(example, candidate, evidence, llm)
    if not post_support.items or any(item.label != "SUPPORTED" for item in post_support.items):
        label = post_support.items[0].label if post_support.items else None
        return no_change("post_support_not_supported", True, resolved, post_suff.answer_sufficiency, label)
    accepted = make_revision(original["draft_id"], feedback_id, answer, explanation)
    return V3RevisionDecision(accepted, True, True, "passed_both_postchecks", resolved,
                              post_suff.answer_sufficiency, "SUPPORTED")
