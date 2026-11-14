"""Phase 2 verifier-only diagnostic; keeps V2 shared contracts and revision untouched.

Run this alongside V2 with fixed question, draft, claims, and retrieved evidence.
This is not a replacement for the existing verify_draft/verify_claims functions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

from ._common import draft_payload, evidence_payload, example_payload, generate_json, text
from .errors import VerificationError
from .verifier import verify_claims, verify_draft

PROMPT_VERSION = "d-answer-sufficiency-v3"
SUFFICIENCY_LABELS = frozenset({"COMPLETE", "INCOMPLETE", "UNCERTAIN"})


@dataclass(frozen=True)
class SufficiencyResult:
    question_id: str
    draft_id: str
    answer_sufficiency: str
    missing_information: str | None
    rationale: str
    evidence_ids: list[str]
    prompt_version: str


@dataclass(frozen=True)
class DualDimensionResult:
    """Plain serializable diagnostic. `claim_feedback` is V2-compatible JSON."""
    claim_feedback: dict[str, Any]
    sufficiency: SufficiencyResult

    def to_dict(self) -> dict[str, Any]:
        return {"claim_feedback": self.claim_feedback, "sufficiency": asdict(self.sufficiency)}


def _feedback_dict(feedback: Any) -> dict[str, Any]:
    return {
        "feedback_id": feedback.feedback_id,
        "items": [{
            "scope": item.scope, "target_id": item.target_id,
            "label": item.label, "rationale": item.rationale,
            "evidence_ids": list(item.evidence_ids),
            "suggested_correction": item.suggested_correction,
        } for item in feedback.items],
        "overall_comment": feedback.overall_comment,
    }


def verify_answer_sufficiency(example: Any, draft: Any, evidence: list[Any], llm: Any) -> SufficiencyResult:
    """Assess whether evidence plus the draft answer resolves the original question.

    This does not adjudicate each claim. Never pass gold answers or gold facts.
    The supplied evidence is the only factual basis, and citations are scoped to it.
    """
    public_example, context = example_payload(example, VerificationError)
    original = draft_payload(draft, VerificationError)
    entries = evidence_payload(evidence, context, VerificationError)
    for i, entry in enumerate(entries, 1):
        entry["citation_id"] = f"S1E{i}"
    lookup = {e["citation_id"]: e["evidence_id"] for e in entries}
    lookup.update({e["evidence_id"]: e["evidence_id"] for e in entries})
    if not entries:
        return SufficiencyResult(public_example["example_id"], original["draft_id"],
                                 "UNCERTAIN", "No retrieved evidence was supplied.",
                                 "Cannot establish answer completeness without evidence.", [], PROMPT_VERSION)
    response = generate_json(llm, "verify_sufficiency", {
        "question": public_example["question"],
        "draft": original,
        "evidence": entries,
        "allowed_evidence_ids": [e["citation_id"] for e in entries],
    }, VerificationError)
    if set(response) != {"answer_sufficiency", "missing_information", "rationale", "evidence_ids"}:
        raise VerificationError("Sufficiency response must contain exactly four expected fields.")
    label = response["answer_sufficiency"]
    if not isinstance(label, str) or label not in SUFFICIENCY_LABELS:
        raise VerificationError("Invalid answer_sufficiency label.")
    missing = response["missing_information"]
    if missing is not None:
        missing = text(missing, "missing_information", VerificationError)
    if label == "COMPLETE" and missing is not None:
        raise VerificationError("COMPLETE must not report missing information.")
    if label == "INCOMPLETE" and missing is None:
        raise VerificationError("INCOMPLETE must specify missing_information.")
    rationale = text(response["rationale"], "rationale", VerificationError)
    citations = response["evidence_ids"]
    if not isinstance(citations, list):
        raise VerificationError("Sufficiency evidence_ids must be a list.")
    resolved: list[str] = []
    for citation in citations:
        if not isinstance(citation, str) or citation not in lookup:
            raise VerificationError("Sufficiency citation is not in supplied evidence.")
        canonical = lookup[citation]
        if canonical not in resolved:
            resolved.append(canonical)
    if label == "COMPLETE" and not resolved:
        raise VerificationError("COMPLETE requires at least one evidence citation.")
    return SufficiencyResult(public_example["example_id"], original["draft_id"],
                             label, missing, rationale, resolved, PROMPT_VERSION)


def verify_dual_dimension(example: Any, draft: Any, claims: list[Any],
                          evidence_by_claim: dict[str, list[Any]], llm: Any) -> DualDimensionResult:
    """V2 per-claim support plus a separate question-level sufficiency call.

    For fair V2/V3 comparison, use the same claims and retrieval evidence. The
    extra call adds cost; capture client usage separately in the runner.
    """
    claim_feedback = verify_claims(example, draft, claims, evidence_by_claim, llm)
    # Preserve original context order, not retrieval ranking; deduplicate by key.
    allowed = set()
    for hits in evidence_by_claim.values():
        for hit in hits:
            allowed.add((hit.doc_title, hit.sentence_id))
    evidence = []
    for sentence in example.context:
        if (sentence.title, sentence.sentence_id) in allowed:
            # Reuse original Evidence instance so score/rank are not fabricated.
            hit = next(h for hits in evidence_by_claim.values() for h in hits
                       if (h.doc_title, h.sentence_id) == (sentence.title, sentence.sentence_id))
            evidence.append(hit)
    sufficiency = verify_answer_sufficiency(example, draft, evidence, llm)
    return DualDimensionResult(_feedback_dict(claim_feedback), sufficiency)


def verify_draft_dual_dimension(example: Any, draft: Any, evidence: list[Any], llm: Any) -> DualDimensionResult:
    """V2 draft support plus separate answer sufficiency for M4 diagnostic."""
    feedback = verify_draft(example, draft, evidence, llm)
    return DualDimensionResult(_feedback_dict(feedback), verify_answer_sufficiency(example, draft, evidence, llm))
