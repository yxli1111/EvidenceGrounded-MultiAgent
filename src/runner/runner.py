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
* Always wraps the injected llm with UsageWrapper for raw_calls.jsonl logging.
  If the inner client already has drain_usage, UsageWrapper delegates to it.

Integration notes
-----------------
* All ports are injected — no import of A/B/C/D modules here.
* Until contracts/schemas.py exists, pass MockQAInput objects and mock ports.
  When real implementations arrive, replace the ports; the runner is unchanged.
* D's configure() must be called by the experiment script BEFORE creating the
  runner, not inside the runner (see evaluation/E_CONTRACT_PROPOSAL.md GAP-6).
* M4 query = draft.explanation (or question + "\\n" + explanation when
  query_include_question=True).  M5 query = claim.claim_text (or question +
  "\\n" + claim_text).  Template is identical for both methods; controlled by
  the query_include_question constructor flag.
"""

from __future__ import annotations

import pathlib
import time
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
from .serialization import (
    ser_claims,
    ser_draft,
    ser_evidence_list,
    ser_feedback,
    ser_revision,
)
from .usage_wrapper import UsageWrapper

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
        methods: list[str] | None = None,
        progress_fn=None,
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
            methods: if given, only run the listed method IDs (e.g. ["M3","M4","M5"]).
                Defaults to None (run all five methods).
            progress_fn: optional callable invoked after every processed example.
                Signature: (idx: int, total: int, example_id: str,
                            example_records: list[ExperimentRecord],
                            elapsed_seconds: float) -> None.

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

        # QAGold isolation: examples must never carry gold labels
        for ex in examples:
            if hasattr(ex, "gold_answer") or (isinstance(ex, dict) and "gold_answer" in ex):
                raise ValueError(
                    "QAGold must not be passed to ExperimentRunner.run(). "
                    "Pass QAInput objects only; keep gold lookup separate."
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

        # Compute raw_calls_path alongside the output file
        raw_calls_path = str(pathlib.Path(output_path).parent / "raw_calls.jsonl")

        # Always wrap llm with UsageWrapper for raw_calls.jsonl logging.
        # If inner already has drain_usage, UsageWrapper delegates to it.
        # This ensures raw_calls.jsonl is always written regardless of the inner client.
        if isinstance(self.llm, UsageWrapper):
            effective_llm = self.llm
            effective_llm.raw_calls_path = raw_calls_path
        else:
            effective_llm = UsageWrapper(self.llm, raw_calls_path=raw_calls_path)

        method_ids = [m for m in _METHOD_IDS if methods is None or m in methods]

        completed = load_completed(output_path, retry_errors=retry_errors)
        dpath = drafts_path(output_path)
        draft_cache = load_draft_cache(dpath)
        new_records: list[ExperimentRecord] = []

        start_time = time.monotonic()
        example_idx = 0

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

            # Skip the whole example if all requested methods are already complete.
            if all((eid, mid) in completed for mid in method_ids):
                continue

            example_new: list[ExperimentRecord] = []

            # Set context for raw call logging
            if hasattr(effective_llm, "set_context"):
                effective_llm.set_context(eid, "draft")

            # Get draft from cache or generate a new one.
            if eid in draft_cache:
                cached = draft_cache[eid]
                draft: Any = _make_draft_obj(cached)
                draft_usage: list[dict] = cached["draft_usage"]
            else:
                try:
                    draft = self.draft_port.generate_draft(example, effective_llm)
                    draft_usage = self._drain_usage(effective_llm)
                except Exception as exc:
                    draft_usage = self._drain_usage(effective_llm)
                    is_fatal = type(exc).__name__ in _FATAL_ERROR_TYPES
                    for mid in method_ids:
                        if (eid, mid) in completed:
                            continue
                        rec = make_error_record(eid, mid, str(exc), question, effective_ms)
                        rec.attempt = attempt_map.get((eid, mid), 0) + 1
                        rec.draft_usage = draft_usage
                        rec.prompt_versions = prompt_versions
                        rec.question_type = question_type
                        rec.raw_outputs = list(getattr(exc, "raw_outputs", None) or []) or None
                        rec.raw_calls_path = raw_calls_path
                        append_record(output_path, rec)
                        example_new.append(rec)
                    new_records.extend(example_new)
                    if progress_fn is not None:
                        progress_fn(
                            example_idx, len(examples), eid, example_new,
                            time.monotonic() - start_time,
                        )
                    example_idx += 1
                    if is_fatal:
                        raise
                    continue

                dd = ser_draft(draft)
                save_draft(dpath, eid, dd, draft_usage)
                draft_cache[eid] = {
                    "example_id": eid,
                    **dd,
                    "draft_usage": draft_usage,
                }

            for mid in method_ids:
                if (eid, mid) in completed:
                    continue
                attempt = attempt_map.get((eid, mid), 0) + 1

                # Set context for raw call logging
                if hasattr(effective_llm, "set_context"):
                    effective_llm.set_context(eid, mid)

                try:
                    method_fn = getattr(self, f"_run_{mid.lower()}")
                    rec = method_fn(example, draft, effective_ms, effective_llm)
                    rec.attempt = attempt
                except Exception as exc:
                    partial_usage = self._drain_usage(effective_llm)
                    rec = make_error_record(eid, mid, str(exc), question, effective_ms)
                    rec.attempt = attempt
                    rec.usage = partial_usage
                    rec.raw_outputs = list(getattr(exc, "raw_outputs", None) or []) or None
                    rec.draft = ser_draft(draft)
                    rec.draft_usage = list(draft_usage)
                    rec.prompt_versions = prompt_versions
                    rec.question_type = question_type
                    rec.raw_calls_path = raw_calls_path
                    append_record(output_path, rec)
                    example_new.append(rec)
                    if type(exc).__name__ in _FATAL_ERROR_TYPES:
                        new_records.extend(example_new)
                        raise
                    continue

                rec.draft_usage = list(draft_usage)
                rec.prompt_versions = prompt_versions
                rec.question_type = question_type
                rec.raw_calls_path = raw_calls_path
                append_record(output_path, rec)
                example_new.append(rec)

            new_records.extend(example_new)
            if progress_fn is not None:
                progress_fn(
                    example_idx, len(examples), eid, example_new,
                    time.monotonic() - start_time,
                )
            example_idx += 1

        return new_records

    # ------------------------------------------------------------------
    # Per-method runners
    # ------------------------------------------------------------------

    def _run_m1(
        self, example: Any, draft: Any, model_config: dict | None, effective_llm: Any
    ) -> ExperimentRecord:
        dd = ser_draft(draft)
        usage = self._drain_usage(effective_llm)
        return ExperimentRecord(
            example_id=_attr(example, "example_id"),
            method_id="M1",
            status="ok",
            timestamp=_now(),
            question=_attr(example, "question", ""),
            draft=dd,
            final=None,    # M1: no RevisionResult — critical per contract §2, §4
            evidence={},   # M1: no retrieval
            usage=usage,
            model_config=model_config,
        )

    def _run_m2(
        self, example: Any, draft: Any, model_config: dict | None, effective_llm: Any
    ) -> ExperimentRecord:
        run_m2_fn = getattr(self.draft_port, "run_m2", None)
        if callable(run_m2_fn):
            output = run_m2_fn(example, draft, effective_llm)
            usage = self._drain_usage(effective_llm)
            # BaselineOutput has .final (RevisionResult | None) and .feedback
            final_obj = getattr(output, "final", None)
            fb_obj = getattr(output, "feedback", None)
            revision = final_obj if final_obj is not None else output
            if fb_obj is not None:
                feedback_dict = ser_feedback(fb_obj)
            else:
                rd_tmp = ser_revision(revision)
                feedback_dict = {
                    "feedback_id": rd_tmp.get("feedback_id", ""),
                    "items": [],
                    "overall_comment": "M2 self-reflection",
                }
        else:
            revision = self.draft_port.self_reflect(example, draft, effective_llm)
            usage = self._drain_usage(effective_llm)
            rd_tmp = ser_revision(revision)
            feedback_dict = {
                "feedback_id": rd_tmp.get("feedback_id", ""),
                "items": [],
                "overall_comment": "M2 self-reflection",
            }
        dd = ser_draft(draft)
        rd = ser_revision(revision)
        return ExperimentRecord(
            example_id=_attr(example, "example_id"),
            method_id="M2",
            status="ok",
            timestamp=_now(),
            question=_attr(example, "question", ""),
            draft=dd,
            final=rd,
            feedback=feedback_dict,
            usage=usage,
            model_config=model_config,
        )

    def _run_m3(
        self, example: Any, draft: Any, model_config: dict | None, effective_llm: Any
    ) -> ExperimentRecord:
        feedback = self.draft_port.independent_critique(example, draft, effective_llm)
        revision = self.verification_port.revise_answer(example, draft, feedback, effective_llm)
        usage = self._drain_usage(effective_llm)
        dd = ser_draft(draft)
        rd = ser_revision(revision)
        return ExperimentRecord(
            example_id=_attr(example, "example_id"),
            method_id="M3",
            status="ok",
            timestamp=_now(),
            question=_attr(example, "question", ""),
            draft=dd,
            final=rd,
            feedback=ser_feedback(feedback),
            usage=usage,
            model_config=model_config,
        )

    def _run_m4(
        self, example: Any, draft: Any, model_config: dict | None, effective_llm: Any
    ) -> ExperimentRecord:
        effective_k = self.m4_top_k if self.m4_top_k is not None else self.top_k
        question = _attr(example, "question", "")
        explanation = _attr(draft, "explanation") or ""
        if isinstance(draft, dict):
            explanation = draft.get("explanation", "")
        query = (question + "\n" + explanation) if self.query_include_question and question else explanation
        context = _attr(example, "context")
        evidence = self.retrieve_port.retrieve(query, context, effective_k)
        feedback = self.verification_port.verify_draft(example, draft, evidence, effective_llm)
        revision = self.verification_port.revise_answer(example, draft, feedback, effective_llm)
        usage = self._drain_usage(effective_llm)
        dd = ser_draft(draft)
        rd = ser_revision(revision)
        return ExperimentRecord(
            example_id=_attr(example, "example_id"),
            method_id="M4",
            status="ok",
            timestamp=_now(),
            question=_attr(example, "question", ""),
            draft=dd,
            final=rd,
            # evidence keyed by draft_id (M4 convention)
            evidence={dd["draft_id"]: ser_evidence_list(evidence)},
            feedback=ser_feedback(feedback),
            usage=usage,
            model_config=model_config,
        )

    def _run_m5(
        self, example: Any, draft: Any, model_config: dict | None, effective_llm: Any
    ) -> ExperimentRecord:
        context = _attr(example, "context")
        question = _attr(example, "question", "")
        claims = self.claims_port.extract_claims(draft, effective_llm)

        top_k_per_claim = self.m5_top_k_per_claim if self.m5_top_k_per_claim is not None else self.top_k
        allocations = _m5_allocations(len(claims), top_k_per_claim, self.m5_total_evidence_budget)

        evidence_by_claim: dict[str, list] = {}
        evidence_by_claim_dicts: dict[str, list[dict]] = {}

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
            ev_dicts = ser_evidence_list(evs)
            evidence_by_claim_dicts[cid] = ev_dicts

        # §9.3: empty claims is a valid result (no factual content). verify_claims still runs.
        # Only exceptions from extract_claims propagate as status="error".
        feedback = self.verification_port.verify_claims(
            example, draft, claims, evidence_by_claim, effective_llm
        )
        revision = self.verification_port.revise_answer(example, draft, feedback, effective_llm)
        usage = self._drain_usage(effective_llm)
        dd = ser_draft(draft)
        rd = ser_revision(revision)
        # evidence keyed by claim_id (M5 convention)
        evidence = {cid: ev_dicts for cid, ev_dicts in evidence_by_claim_dicts.items()}
        return ExperimentRecord(
            example_id=_attr(example, "example_id"),
            method_id="M5",
            status="ok",
            timestamp=_now(),
            question=_attr(example, "question", ""),
            draft=dd,
            final=rd,
            claims=ser_claims(claims),
            evidence=evidence,
            feedback=ser_feedback(feedback),
            usage=usage,
            model_config=model_config,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drain_usage(self, effective_llm: Any) -> list[dict]:
        drain_fn = getattr(effective_llm, "drain_usage", None)
        if callable(drain_fn):
            return drain_fn()
        return []
