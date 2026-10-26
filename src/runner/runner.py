"""ExperimentRunner: orchestrates M1–M5 for every QA example.

Responsibilities
----------------
* Generate one shared AnswerDraft per example and reuse it for all five methods
  (CONTRACT Section 5: "Use the same QA examples and shared draft").
* Write each ExperimentRecord to JSONL immediately after completion so that a
  crash or interruption can be resumed without re-running completed pairs.
* On resume, skip any (example_id, method_id) pair already present in the file.
* Record status="error" for any method that raises; the example still appears in
  every metric denominator (CONTRACT: "A failed method must not silently disappear").
* Collect LLM usage from each method run via llm.drain_usage() if available.
  draft_usage (generate_draft) is kept separate to avoid 5× double-counting in
  cost analysis; P3 decides how to allocate shared draft cost.

Integration notes
-----------------
* All ports are injected — no import of A/B/C/D modules here.
* Until contracts/schemas.py exists, pass MockQAInput objects and mock ports.
  When real implementations arrive, replace the ports; the runner is unchanged.
* D's configure() must be called by the experiment script BEFORE creating the
  runner, not inside the runner (see evaluation/E_CONTRACT_PROPOSAL.md GAP-6).
* M4 query = draft.explanation (plain explanation without question prefix).
  M5 query = claim.claim_text.  Both policies are frozen per CONTRACT pending
  a team decision; change the _m4_query / _m5_query helpers if the team decides
  to add the question prefix.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .records import (
    ExperimentRecord,
    append_record,
    load_completed,
    make_error_record,
)

_METHOD_IDS = ["M1", "M2", "M3", "M4", "M5"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ---------------------------------------------------------------------------
# Evidence / feedback serialisers (work with both dicts and mock/real objects)
# ---------------------------------------------------------------------------

def _ser_draft(d: Any) -> dict:
    if isinstance(d, dict):
        return d
    return {
        "draft_id": d.draft_id,
        "short_answer": d.short_answer,
        "explanation": d.explanation,
    }


def _ser_revision(r: Any) -> dict:
    if isinstance(r, dict):
        return r
    return {
        "draft_id": r.draft_id,
        "feedback_id": r.feedback_id,
        "final_answer": r.final_answer,
        "final_explanation": r.final_explanation,
    }


def _ser_feedback(fb: Any) -> dict:
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


def _ser_evidence_list(evs: list) -> list[dict]:
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


def _ser_claims(claims: list) -> list[dict]:
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


def _m4_query(draft: Any) -> str:
    return _attr(draft, "explanation") or ""


def _m5_query(claim: Any) -> str:
    return _attr(claim, "claim_text") or ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """Orchestrates M1–M5 for a list of QA examples.

    Args:
        draft_port: A's generate_draft / self_reflect / independent_critique.
        retrieve_port: B's retrieve.
        claims_port: C's extract_claims.
        verification_port: D's verify_draft / verify_claims / revise_answer.
        llm: C's LLMClient (injected into every port call).
        top_k: number of evidence sentences to retrieve per query.
    """

    def __init__(
        self,
        draft_port: Any,
        retrieve_port: Any,
        claims_port: Any,
        verification_port: Any,
        llm: Any,
        top_k: int = 5,
    ) -> None:
        self.draft_port = draft_port
        self.retrieve_port = retrieve_port
        self.claims_port = claims_port
        self.verification_port = verification_port
        self.llm = llm
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        examples: list,
        output_path: str,
        model_settings: dict | None = None,
    ) -> list[ExperimentRecord]:
        """Run all uncompleted (example, method) pairs and append to output_path.

        Args:
            examples: list of QAInput (or mock) objects.
            output_path: path to JSONL file; created if absent, appended if present.
            model_settings: dict snapshot of model/sampling config stored in each record.

        Returns:
            List of newly created ExperimentRecords (does not include pre-existing ones).
        """
        completed = load_completed(output_path)
        new_records: list[ExperimentRecord] = []

        for example in examples:
            eid = _attr(example, "example_id")
            question = _attr(example, "question", "")

            # Generate one shared draft for all five methods.
            try:
                draft = self.draft_port.generate_draft(example, self.llm)
                draft_usage = self._drain_usage()
            except Exception as exc:
                draft_usage = self._drain_usage()
                for mid in _METHOD_IDS:
                    if (eid, mid) in completed:
                        continue
                    rec = make_error_record(eid, mid, str(exc), question, model_settings)
                    rec.draft_usage = draft_usage
                    append_record(output_path, rec)
                    new_records.append(rec)
                continue

            for mid in _METHOD_IDS:
                if (eid, mid) in completed:
                    continue
                try:
                    method_fn = getattr(self, f"_run_{mid.lower()}")
                    rec = method_fn(example, draft, model_settings)
                except Exception as exc:
                    self._drain_usage()  # discard partial usage on error
                    rec = make_error_record(eid, mid, str(exc), question, model_settings)

                rec.draft_usage = list(draft_usage)
                append_record(output_path, rec)
                new_records.append(rec)

        return new_records

    # ------------------------------------------------------------------
    # Per-method runners
    # ------------------------------------------------------------------

    def _run_m1(
        self, example: Any, draft: Any, model_settings: dict | None
    ) -> ExperimentRecord:
        dd = _ser_draft(draft)
        usage = self._drain_usage()
        return ExperimentRecord(
            example_id=_attr(example, "example_id"),
            method_id="M1",
            status="ok",
            timestamp=_now(),
            question=_attr(example, "question", ""),
            draft_id=dd["draft_id"],
            short_answer=dd["short_answer"],
            explanation=dd["explanation"],
            final_answer=dd["short_answer"],        # M1: draft is final
            final_explanation=dd["explanation"],
            usage=usage,
            model_settings=model_settings,
        )

    def _run_m2(
        self, example: Any, draft: Any, model_settings: dict | None
    ) -> ExperimentRecord:
        revision = self.draft_port.self_reflect(example, draft, self.llm)
        usage = self._drain_usage()
        dd = _ser_draft(draft)
        rd = _ser_revision(revision)
        return ExperimentRecord(
            example_id=_attr(example, "example_id"),
            method_id="M2",
            status="ok",
            timestamp=_now(),
            question=_attr(example, "question", ""),
            draft_id=dd["draft_id"],
            short_answer=dd["short_answer"],
            explanation=dd["explanation"],
            final_answer=rd["final_answer"],
            final_explanation=rd["final_explanation"],
            feedback={"feedback_id": rd["feedback_id"], "items": [], "overall_comment": "M2 self-reflection"},
            usage=usage,
            model_settings=model_settings,
        )

    def _run_m3(
        self, example: Any, draft: Any, model_settings: dict | None
    ) -> ExperimentRecord:
        feedback = self.draft_port.independent_critique(example, draft, self.llm)
        revision = self.verification_port.revise_answer(example, draft, feedback, self.llm)
        usage = self._drain_usage()
        dd = _ser_draft(draft)
        rd = _ser_revision(revision)
        return ExperimentRecord(
            example_id=_attr(example, "example_id"),
            method_id="M3",
            status="ok",
            timestamp=_now(),
            question=_attr(example, "question", ""),
            draft_id=dd["draft_id"],
            short_answer=dd["short_answer"],
            explanation=dd["explanation"],
            final_answer=rd["final_answer"],
            final_explanation=rd["final_explanation"],
            feedback=_ser_feedback(feedback),
            usage=usage,
            model_settings=model_settings,
        )

    def _run_m4(
        self, example: Any, draft: Any, model_settings: dict | None
    ) -> ExperimentRecord:
        query = _m4_query(draft)
        context = _attr(example, "context")
        evidence = self.retrieve_port.retrieve(query, context, self.top_k)
        feedback = self.verification_port.verify_draft(example, draft, evidence, self.llm)
        revision = self.verification_port.revise_answer(example, draft, feedback, self.llm)
        usage = self._drain_usage()
        dd = _ser_draft(draft)
        rd = _ser_revision(revision)
        return ExperimentRecord(
            example_id=_attr(example, "example_id"),
            method_id="M4",
            status="ok",
            timestamp=_now(),
            question=_attr(example, "question", ""),
            draft_id=dd["draft_id"],
            short_answer=dd["short_answer"],
            explanation=dd["explanation"],
            final_answer=rd["final_answer"],
            final_explanation=rd["final_explanation"],
            evidence=_ser_evidence_list(evidence),
            feedback=_ser_feedback(feedback),
            usage=usage,
            model_settings=model_settings,
        )

    def _run_m5(
        self, example: Any, draft: Any, model_settings: dict | None
    ) -> ExperimentRecord:
        context = _attr(example, "context")
        claims = self.claims_port.extract_claims(draft, self.llm)

        evidence_by_claim: dict[str, list] = {}
        evidence_by_claim_dicts: dict[str, list[dict]] = {}
        all_ev_dicts: list[dict] = []
        seen: set[tuple] = set()

        for claim in claims:
            cid = _attr(claim, "claim_id")
            evs = self.retrieve_port.retrieve(_m5_query(claim), context, self.top_k)
            evidence_by_claim[cid] = evs
            ev_dicts = _ser_evidence_list(evs)
            evidence_by_claim_dicts[cid] = ev_dicts
            for ev in ev_dicts:
                key = (ev["doc_title"], ev["sentence_id"])
                if key not in seen:
                    seen.add(key)
                    all_ev_dicts.append(ev)

        feedback = self.verification_port.verify_claims(
            example, draft, claims, evidence_by_claim, self.llm
        )
        revision = self.verification_port.revise_answer(example, draft, feedback, self.llm)
        usage = self._drain_usage()
        dd = _ser_draft(draft)
        rd = _ser_revision(revision)
        return ExperimentRecord(
            example_id=_attr(example, "example_id"),
            method_id="M5",
            status="ok",
            timestamp=_now(),
            question=_attr(example, "question", ""),
            draft_id=dd["draft_id"],
            short_answer=dd["short_answer"],
            explanation=dd["explanation"],
            final_answer=rd["final_answer"],
            final_explanation=rd["final_explanation"],
            claims=_ser_claims(claims),
            evidence=all_ev_dicts,
            evidence_by_claim=evidence_by_claim_dicts,
            feedback=_ser_feedback(feedback),
            usage=usage,
            model_settings=model_settings,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_usage(self) -> list[dict]:
        drain_fn = getattr(self.llm, "drain_usage", None)
        if callable(drain_fn):
            return drain_fn()
        return []
