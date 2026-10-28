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
* M4 query = draft.explanation (or question + "\n" + explanation when
  query_include_question=True).  M5 query = claim.claim_text (or question +
  "\n" + claim_text).  Template is identical for both methods; controlled by
  the query_include_question constructor flag.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .records import (
    ExperimentRecord,
    append_record,
    drafts_path,
    load_completed,
    load_draft_cache,
    load_records,
    make_error_record,
    save_draft,
)

_METHOD_IDS = ["M1", "M2", "M3", "M4", "M5"]

# Errors that should abort the entire run (not just one example)
_FATAL_ERROR_TYPES: frozenset[str] = frozenset({
    "LLMAuthenticationError",
    "LLMInsufficientCreditError",
    "ConfigurationError",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _collect_prompt_versions(ports: list[tuple[str, Any]]) -> dict:
    """Merge prompt_versions from all ports into one dict, skipping None values.

    If a port has a .prompt_versions dict, all non-None entries are merged in.
    If it only has .prompt_version (str), it is stored under the port name
    only when the value is not None.
    """
    result: dict = {}
    for port_name, port in ports:
        pv = getattr(port, "prompt_versions", None)
        if isinstance(pv, dict):
            result.update({k: v for k, v in pv.items() if v is not None})
        else:
            val = getattr(port, "prompt_version", None)
            if val is not None:
                result[port_name] = val
    return result


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


def _make_draft_obj(d: dict) -> Any:
    """Convert a cached draft dict to an AnswerDraft object.

    Tries contracts.schemas.AnswerDraft first; falls back to _DraftFallback
    so that ports doing draft.draft_id / draft.explanation always work.
    """
    try:
        from contracts.schemas import AnswerDraft
    except ImportError:
        from .records import _DraftFallback as AnswerDraft  # type: ignore[assignment]
    return AnswerDraft(
        draft_id=d["draft_id"],
        short_answer=d["short_answer"],
        explanation=d["explanation"],
    )


def _m5_allocations(
    n_claims: int, top_k_per_claim: int, budget: int | None
) -> list[int]:
    """Round-robin evidence budget allocation across claims.

    If budget is None, every claim gets top_k_per_claim slots.
    Otherwise slots are distributed round-robin then capped at top_k_per_claim.
    D requires every claim_id in evidence_by_claim; claims with 0 slots get [].
    """
    if budget is None or n_claims == 0:
        return [top_k_per_claim] * n_claims
    allocations = [0] * n_claims
    for slot in range(budget):
        allocations[slot % n_claims] += 1
    return [min(a, top_k_per_claim) for a in allocations]


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
        top_k: default evidence sentences per retrieval call (M4 and M5 fallback).
        m4_top_k: evidence sentences for M4 (overrides top_k; None = use top_k).
        m5_top_k_per_claim: evidence sentences per claim in M5 (None = use top_k).
        m5_total_evidence_budget: total evidence slots across all M5 claims;
            distributed round-robin.  None = no limit.
        query_include_question: if True, prepend question + "\\n" to M4/M5 queries.
            Default False (explanation / claim_text only).
    """

    def __init__(
        self,
        draft_port: Any,
        retrieve_port: Any,
        claims_port: Any,
        verification_port: Any,
        llm: Any,
        top_k: int = 5,
        m4_top_k: int | None = None,
        m5_top_k_per_claim: int | None = None,
        m5_total_evidence_budget: int | None = None,
        query_include_question: bool = False,
    ) -> None:
        self.draft_port = draft_port
        self.retrieve_port = retrieve_port
        self.claims_port = claims_port
        self.verification_port = verification_port
        self.llm = llm
        self.top_k = top_k
        self.m4_top_k = m4_top_k
        self.m5_top_k_per_claim = m5_top_k_per_claim
        self.m5_total_evidence_budget = m5_total_evidence_budget
        self.query_include_question = query_include_question

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        examples: list,
        output_path: str,
        model_settings: dict | None = None,
        exclude_ids: set[str] | None = None,
        retry_errors: bool = False,
    ) -> list[ExperimentRecord]:
        """Run all uncompleted (example, method) pairs and append to output_path.

        Args:
            examples: list of QAInput (or mock) objects.
            output_path: path to JSONL file; created if absent, appended if present.
            model_settings: dict snapshot of model/sampling config stored in each record.
            exclude_ids: set of example_ids that must not be in examples
                (e.g. dev IDs excluded from formal evaluation).  Raises ValueError
                if any overlap is found — the caller must filter before running.
            retry_errors: if True, re-run (example_id, method_id) pairs whose latest
                attempt has status="error".  New records are appended with
                attempt = previous_max + 1.  Old error records are never modified.

        Returns:
            List of newly created ExperimentRecords (does not include pre-existing ones).
        """
        if exclude_ids:
            overlap = {
                _attr(ex, "example_id") for ex in examples
            } & set(exclude_ids)
            if overlap:
                raise ValueError(
                    f"Examples in exclude_ids must be removed before running. "
                    f"Overlapping IDs: {sorted(overlap)}"
                )

        # Merge retrieval params into a copy of model_settings for full traceability.
        effective_ms: dict = dict(model_settings) if model_settings else {}
        effective_ms["_retrieval"] = {
            "m4_top_k": self.m4_top_k if self.m4_top_k is not None else self.top_k,
            "m5_top_k_per_claim": self.m5_top_k_per_claim if self.m5_top_k_per_claim is not None else self.top_k,
            "m5_total_evidence_budget": self.m5_total_evidence_budget,
            "query_include_question": self.query_include_question,
            "query_template": "question+nl+text" if self.query_include_question else "text_only",
        }

        completed = load_completed(output_path, retry_errors=retry_errors)
        dpath = drafts_path(output_path)
        draft_cache = load_draft_cache(dpath)
        new_records: list[ExperimentRecord] = []

        # Pre-compute max attempt number per (example_id, method_id) for retry tracking.
        attempt_map: dict[tuple, int] = {}
        if retry_errors:
            for rec in load_records(output_path):
                key = (rec.example_id, rec.method_id)
                attempt_map[key] = max(attempt_map.get(key, 0), rec.attempt)

        prompt_versions = _collect_prompt_versions([
            ("draft_port", self.draft_port),
            ("retrieve_port", self.retrieve_port),
            ("claims_port", self.claims_port),
            ("verification_port", self.verification_port),
        ])

        for example in examples:
            eid = _attr(example, "example_id")
            question = _attr(example, "question", "")
            question_type = _attr(example, "question_type")

            # Skip the whole example if all 5 methods are already complete.
            if all((eid, mid) in completed for mid in _METHOD_IDS):
                continue

            # Get draft from cache or generate a new one.
            if eid in draft_cache:
                cached = draft_cache[eid]
                draft: Any = _make_draft_obj(cached)
                draft_usage: list[dict] = cached["draft_usage"]
            else:
                try:
                    draft = self.draft_port.generate_draft(example, self.llm)
                    draft_usage = self._drain_usage()
                except Exception as exc:
                    draft_usage = self._drain_usage()
                    is_fatal = type(exc).__name__ in _FATAL_ERROR_TYPES
                    for mid in _METHOD_IDS:
                        if (eid, mid) in completed:
                            continue
                        rec = make_error_record(eid, mid, str(exc), question, effective_ms)
                        rec.attempt = attempt_map.get((eid, mid), 0) + 1
                        rec.draft_usage = draft_usage
                        rec.prompt_versions = prompt_versions
                        rec.question_type = question_type
                        rec.raw_outputs = list(getattr(exc, "raw_outputs", None) or []) or None
                        append_record(output_path, rec)
                        new_records.append(rec)
                    if is_fatal:
                        raise
                    continue

                dd = _ser_draft(draft)
                save_draft(dpath, eid, dd, draft_usage)
                draft_cache[eid] = {
                    "example_id": eid,
                    **dd,
                    "draft_usage": draft_usage,
                }

            for mid in _METHOD_IDS:
                if (eid, mid) in completed:
                    continue
                attempt = attempt_map.get((eid, mid), 0) + 1
                try:
                    method_fn = getattr(self, f"_run_{mid.lower()}")
                    rec = method_fn(example, draft, effective_ms)
                    rec.attempt = attempt
                except Exception as exc:
                    partial_usage = self._drain_usage()
                    rec = make_error_record(eid, mid, str(exc), question, effective_ms)
                    rec.attempt = attempt
                    rec.usage = partial_usage
                    rec.raw_outputs = list(getattr(exc, "raw_outputs", None) or []) or None
                    rec.draft_usage = list(draft_usage)
                    rec.prompt_versions = prompt_versions
                    rec.question_type = question_type
                    append_record(output_path, rec)
                    new_records.append(rec)
                    if type(exc).__name__ in _FATAL_ERROR_TYPES:
                        raise
                    continue

                rec.draft_usage = list(draft_usage)
                rec.prompt_versions = prompt_versions
                rec.question_type = question_type
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
        run_m2_fn = getattr(self.draft_port, "run_m2", None)
        if callable(run_m2_fn):
            output = run_m2_fn(example, draft, self.llm)
            usage = self._drain_usage()
            # BaselineOutput has .final (RevisionResult | None) and .feedback
            final_obj = getattr(output, "final", None)
            fb_obj = getattr(output, "feedback", None)
            revision = final_obj if final_obj is not None else output
            if fb_obj is not None:
                feedback_dict = _ser_feedback(fb_obj)
            else:
                rd_tmp = _ser_revision(revision)
                feedback_dict = {
                    "feedback_id": rd_tmp.get("feedback_id", ""),
                    "items": [],
                    "overall_comment": "M2 self-reflection",
                }
        else:
            revision = self.draft_port.self_reflect(example, draft, self.llm)
            usage = self._drain_usage()
            rd_tmp = _ser_revision(revision)
            feedback_dict = {
                "feedback_id": rd_tmp.get("feedback_id", ""),
                "items": [],
                "overall_comment": "M2 self-reflection",
            }
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
            feedback=feedback_dict,
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
        effective_k = self.m4_top_k if self.m4_top_k is not None else self.top_k
        question = _attr(example, "question", "")
        explanation = _attr(draft, "explanation") or ""
        if isinstance(draft, dict):
            explanation = draft.get("explanation", "")
        query = (question + "\n" + explanation) if self.query_include_question and question else explanation
        context = _attr(example, "context")
        evidence = self.retrieve_port.retrieve(query, context, effective_k)
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
        question = _attr(example, "question", "")
        claims = self.claims_port.extract_claims(draft, self.llm)

        top_k_per_claim = self.m5_top_k_per_claim if self.m5_top_k_per_claim is not None else self.top_k
        allocations = _m5_allocations(len(claims), top_k_per_claim, self.m5_total_evidence_budget)

        evidence_by_claim: dict[str, list] = {}
        evidence_by_claim_dicts: dict[str, list[dict]] = {}
        all_ev_dicts: list[dict] = []
        seen: set[tuple] = set()

        for i, claim in enumerate(claims):
            cid = _attr(claim, "claim_id")
            claim_text = _attr(claim, "claim_text") or ""
            claim_query = (question + "\n" + claim_text) if self.query_include_question and question else claim_text
            alloc_k = allocations[i]
            if alloc_k > 0:
                evs = self.retrieve_port.retrieve(claim_query, context, alloc_k)
            else:
                evs = []
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
