"""M4 whole-draft and M5 atomic-claim evidence verification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import (
    LABELS, draft_payload, evidence_payload, example_payload, feedback_result,
    generate_json, sequence, text,
)
from .errors import VerificationError
from .schema_adapter import feedback_schema

if TYPE_CHECKING:
    from contracts.schemas import AnswerDraft, AtomicClaim, Evidence, QAInput, VerificationFeedback
    from contracts.interfaces import LLMClient


def _verify(example, draft, targets: list[dict], scope: str, stage: str, llm):
    feedback_schema()
    original = draft_payload(draft, VerificationError)
    if not targets:
        return feedback_result(original["draft_id"], stage, [], "No factual claims to verify.")
    if all(not target["evidence"] for target in targets):
        items = [{
            "scope": scope,
            "target_id": target["target_id"],
            "label": "UNSUPPORTED",
            "rationale": "No retrieved evidence was supplied; this does not establish that the statement is false.",
            "evidence_ids": [],
            "suggested_correction": None,
        } for target in targets]
        return feedback_result(original["draft_id"], stage, items, "Insufficient retrieved evidence.")

    data = generate_json(llm, stage, {
        "question": text(example.question, "question", VerificationError),
        "draft": original,
        "targets": targets,
    }, VerificationError)
    returned = sequence(data.get("items"), "items", VerificationError)
    comment = text(data.get("overall_comment"), "overall_comment", VerificationError, allow_empty=True)
    expected = {target["target_id"]: target for target in targets}
    items_by_id = {}
    for item in returned:
        if not isinstance(item, dict):
            raise VerificationError("Each feedback item must be a JSON object.")
        target_id = text(item.get("target_id"), "target_id", VerificationError)
        if target_id not in expected or target_id in items_by_id:
            raise VerificationError("Unknown or duplicate feedback target_id.")
        label = item.get("label")
        if not isinstance(label, str) or label not in LABELS:
            raise VerificationError("Verifier label must be SUPPORTED, CONTRADICTED or UNSUPPORTED.")
        citations = sequence(item.get("evidence_ids"), "evidence_ids", VerificationError)
        allowed = {entry["evidence_id"] for entry in expected[target_id]["evidence"]}
        for citation in citations:
            if not isinstance(citation, str) or citation not in allowed:
                raise VerificationError("Citation is not in this target's retrieved evidence.")
        if len(set(citations)) != len(citations):
            raise VerificationError("Duplicate evidence citation.")
        if label in {"SUPPORTED", "CONTRADICTED"} and not citations:
            raise VerificationError("SUPPORTED and CONTRADICTED require evidence citations.")
        correction = item.get("suggested_correction")
        if correction is not None:
            correction = text(correction, "suggested_correction", VerificationError)
        if not allowed and (label != "UNSUPPORTED" or correction is not None):
            raise VerificationError("An evidence-free target must be UNSUPPORTED without a correction.")
        items_by_id[target_id] = {
            "scope": scope,
            "target_id": target_id,
            "label": label,
            "rationale": text(item.get("rationale"), "rationale", VerificationError),
            "evidence_ids": citations,
            "suggested_correction": correction,
        }
    if set(items_by_id) != set(expected):
        raise VerificationError("Verifier must return exactly one item per supplied target.")
    return feedback_result(
        original["draft_id"], stage,
        [items_by_id[target["target_id"]] for target in targets], comment,
    )


def verify_draft(
    example: QAInput, draft: AnswerDraft, evidence: list[Evidence], llm: LLMClient
) -> VerificationFeedback:
    """M4: one whole-draft verdict; only the retrieved evidence supports a verdict."""
    _, context = example_payload(example, VerificationError)
    original = draft_payload(draft, VerificationError)
    targets = [{
        "target_id": original["draft_id"],
        "text": original["explanation"],
        "evidence": evidence_payload(evidence, context, VerificationError),
    }]
    return _verify(example, draft, targets, "draft", "verify_draft", llm)


def verify_claims(
    example: QAInput, draft: AnswerDraft, claims: list[AtomicClaim],
    evidence_by_claim: dict[str, list[Evidence]], llm: LLMClient,
) -> VerificationFeedback:
    """M5: one batched call, with evidence kept separate for each claim.

    Every claim must have a mapping entry; [] explicitly means no retrieval hits.
    Missing entries are integration errors, not evidence-insufficiency verdicts.
    """
    _, context = example_payload(example, VerificationError)
    if not isinstance(evidence_by_claim, dict):
        raise VerificationError("evidence_by_claim must be a dict keyed by claim_id.")
    targets = []
    seen = set()
    for claim in sequence(claims, "claims", VerificationError):
        claim_id = text(claim.claim_id, "claim_id", VerificationError)
        if claim_id in seen:
            raise VerificationError("Duplicate input claim_id.")
        seen.add(claim_id)
        if claim_id not in evidence_by_claim:
            raise VerificationError("Missing evidence mapping entry for a claim.")
        targets.append({
            "target_id": claim_id,
            "text": text(claim.claim_text, "claim_text", VerificationError),
            "evidence": evidence_payload(evidence_by_claim[claim_id], context, VerificationError),
        })
    if set(evidence_by_claim) != seen:
        raise VerificationError("Evidence mapping contains an unknown claim_id.")
    return _verify(example, draft, targets, "claim", "verify_claims", llm)
