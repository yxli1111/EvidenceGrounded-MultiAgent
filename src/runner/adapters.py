"""Adapters that bridge the real A/B/C/D implementations to E's runner.

All imports from contracts, A, B, C, D are lazy (inside functions/methods)
so that E's own tests pass without those packages installed.

Classes
-------
BRetrievalAdapter    — wraps B's EvidenceRetriever; converts Sentence ↔ dict and
                       converts B's result dicts → contracts.schemas.Evidence objects
                       (required by D's evidence_payload attribute access)
DVerificationAdapter — wraps D's TWO modules: src.verification (configure/verify_*/
                       runtime_metadata) and src.agents.revision (revise_answer)
AAnswerAdapter       — wraps A's AnswerAgent; exposes run_m2 with correct signature
                       (example, draft, llm, config) from src.agents.baselines;
                       run_m2 is absent when baselines is not importable

Helpers
-------
sentences_to_context_dicts(sentences)  — Sentence objects → list[dict]
b_gold_to_contracts(facts)             — list of dicts or tuples → list[tuple]
b_example_to_contracts(b_dict)         — raw B example dict → (id, question, context)
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Context / sentence conversion
# ---------------------------------------------------------------------------

def sentences_to_context_dicts(context: list) -> list[dict]:
    """Convert Sentence objects (or dicts) to {title, sentence_id, text} dicts.

    B's EvidenceRetriever.retrieve() expects a list[dict].  E's runner works
    with Sentence objects from contracts.schemas.  This helper normalises both.
    """
    result = []
    for item in context:
        if isinstance(item, dict):
            result.append(item)
        else:
            result.append({
                "title": item.title,
                "sentence_id": item.sentence_id,
                "text": item.text,
            })
    return result


def b_gold_to_contracts(facts: list) -> list[tuple]:
    """Convert B's gold_supporting_facts list to contracts tuple format.

    Accepts either:
      - list[dict]: each dict has "title" and "sentence_id" keys
      - list[tuple]: each tuple is (title, sentence_id)
      - mixed list of both

    Returns list[tuple[str, int]].
    """
    result = []
    for f in facts:
        if isinstance(f, tuple):
            result.append((f[0], f[1]))
        else:
            result.append((f["title"], f["sentence_id"]))
    return result


def b_example_to_contracts(b_dict: dict) -> tuple[str, str, list[dict]]:
    """Extract (example_id, question, context_dicts) from B's example dict."""
    return (
        b_dict.get("example_id", ""),
        b_dict.get("question", ""),
        b_dict.get("context", []),
    )


# ---------------------------------------------------------------------------
# BRetrievalAdapter
# ---------------------------------------------------------------------------

class BRetrievalAdapter:
    """Adapts B's OLD dict-based EvidenceRetriever to E's RetrievePort interface.

    Use this adapter ONLY for B's original `src.retrieval.retriever.EvidenceRetriever`
    which expects context as list[dict] with "title" key and returns list[dict].

    Do NOT wrap B's newer ContractEvidenceRetriever (src.retrieval.contract_adapter).
    ContractEvidenceRetriever already accepts list[Sentence] and returns
    list[contracts.schemas.Evidence] objects — it can be passed directly as
    runner's retrieve_port without any adapter.

    This adapter:
    1. Converts Sentence objects / dicts → {title, sentence_id, text} dicts for B
    2. Converts B's result dicts → contracts.schemas.Evidence objects for D

    The Evidence.text is copied verbatim from B's result so that D's validation
    (which checks text == context sentence text) passes without modification.
    """

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever

    def retrieve(self, query: str, context: Any, top_k: int) -> list:
        context_dicts = sentences_to_context_dicts(list(context) if context else [])
        raw = self._retriever.retrieve(query, context_dicts, top_k)
        return [self._to_evidence(ev) for ev in raw]

    @staticmethod
    def _to_evidence(ev: Any) -> Any:
        """Convert B's result dict to contracts.schemas.Evidence object."""
        from contracts.schemas import Evidence  # lazy import
        if isinstance(ev, Evidence):
            return ev
        if isinstance(ev, dict):
            return Evidence(
                doc_title=ev["doc_title"],
                sentence_id=ev["sentence_id"],
                text=ev["text"],
                score=float(ev["score"]),
                retrieval_query=ev.get("retrieval_query", ""),
                rank=ev["rank"],
            )
        # Already an object with the right attributes — return as-is if it is Evidence
        return Evidence(
            doc_title=ev.doc_title,
            sentence_id=ev.sentence_id,
            text=ev.text,
            score=float(ev.score),
            retrieval_query=getattr(ev, "retrieval_query", ""),
            rank=ev.rank,
        )


# ---------------------------------------------------------------------------
# DVerificationAdapter
# ---------------------------------------------------------------------------

class DVerificationAdapter:
    """Adapts D's two-module structure to E's VerificationPort interface.

    D's public API is split across two modules (confirmed from feat/d-verification-revision):
      src.verification        — configure, runtime_metadata, verify_draft, verify_claims
      src.agents.revision     — revise_answer

    Calls verification_module.configure() exactly once at construction (GAP-6).
    prompt_versions comes from verification_module.runtime_metadata() if available.
    """

    def __init__(
        self,
        verification_module: Any,
        revision_module: Any,
        **configure_kwargs: Any,
    ) -> None:
        self._verification = verification_module
        self._revision = revision_module

        configure_fn = getattr(verification_module, "configure", None)
        if callable(configure_fn):
            configure_fn(**configure_kwargs)

        meta_fn = getattr(verification_module, "runtime_metadata", None)
        if callable(meta_fn):
            meta = meta_fn() or {}
            self._prompt_versions: dict = meta.get("prompt_versions", {}) or {}
        else:
            self._prompt_versions = {}

    @property
    def prompt_versions(self) -> dict:
        return self._prompt_versions

    def verify_draft(self, example: Any, draft: Any, evidence: list, llm: Any) -> Any:
        return self._verification.verify_draft(example, draft, evidence, llm)

    def verify_claims(
        self,
        example: Any,
        draft: Any,
        claims: list,
        evidence_by_claim: dict,
        llm: Any,
    ) -> Any:
        return self._verification.verify_claims(example, draft, claims, evidence_by_claim, llm)

    def revise_answer(self, example: Any, draft: Any, feedback: Any, llm: Any) -> Any:
        return self._revision.revise_answer(example, draft, feedback, llm)


# ---------------------------------------------------------------------------
# AAnswerAdapter
# ---------------------------------------------------------------------------

class AAnswerAdapter:
    """Adapts A's AnswerAgent to E's DraftPort interface.

    Exposes generate_draft, self_reflect, independent_critique directly from
    the agent.

    run_m2 is exposed only when src.agents.baselines can be imported.  It calls
    baselines.run_m2(example, draft, llm, agent.config) — config is required.
    There is NO silent fallback to self_reflect when baselines is unavailable.

    prompt_versions is a dict (from agent.prompt_versions).
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent
        # Try to import baselines once at construction time.
        # If successful, bind run_m2 as an instance method.
        try:
            from src.agents import baselines as _baselines  # lazy import
            _config = getattr(agent, "config", None)

            def run_m2(example: Any, draft: Any, llm: Any) -> Any:
                import sys as _sys
                mod = _sys.modules.get("src.agents.baselines", _baselines)
                return mod.run_m2(example, draft, llm, _config)

            self.run_m2 = run_m2  # bound only when baselines is importable
        except (ImportError, AttributeError):
            pass  # run_m2 not provided — caller checks with hasattr

    @property
    def prompt_versions(self) -> dict:
        pv = getattr(self._agent, "prompt_versions", None)
        if isinstance(pv, dict):
            return pv
        return {}

    def generate_draft(self, example: Any, llm: Any) -> Any:
        return self._agent.generate_draft(example, llm)

    def self_reflect(self, example: Any, draft: Any, llm: Any) -> Any:
        return self._agent.self_reflect(example, draft, llm)

    def independent_critique(self, example: Any, draft: Any, llm: Any) -> Any:
        return self._agent.independent_critique(example, draft, llm)
