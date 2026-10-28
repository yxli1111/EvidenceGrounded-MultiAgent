"""Adapters that bridge the real A/B/C/D implementations to E's runner.

All imports from contracts, A, B, C, D are lazy (inside functions/methods)
so that E's own tests pass without those packages installed.

Classes
-------
BRetrievalAdapter   — wraps B's EvidenceRetriever; converts Sentence ↔ dict
DVerificationAdapter — wraps D's module functions; calls configure() on init
AAnswerAdapter       — wraps A's AnswerAgent; exposes run_m2

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
    """Adapts B's EvidenceRetriever to E's RetrievePort interface.

    B's retriever expects context as list[dict] with "title" key.
    E's runner passes Sentence objects (or dicts).  This adapter normalises
    context before passing it to B and returns plain dicts as evidence.

    The Evidence text field is copied verbatim from B's result so that D's
    evidence_payload validation (which compares text against the original
    context sentences) passes without modification.
    """

    def __init__(self, retriever: Any) -> None:
        self._retriever = retriever

    def retrieve(self, query: str, context: Any, top_k: int) -> list[dict]:
        context_dicts = sentences_to_context_dicts(list(context) if context else [])
        raw = self._retriever.retrieve(query, context_dicts, top_k)
        return [self._normalise_evidence(ev) for ev in raw]

    @staticmethod
    def _normalise_evidence(ev: Any) -> dict:
        if isinstance(ev, dict):
            return ev
        return {
            "doc_title": ev.doc_title,
            "sentence_id": ev.sentence_id,
            "text": ev.text,
            "score": float(ev.score),
            "retrieval_query": ev.retrieval_query,
            "rank": ev.rank,
        }


# ---------------------------------------------------------------------------
# DVerificationAdapter
# ---------------------------------------------------------------------------

class DVerificationAdapter:
    """Adapts D's module-level functions to E's VerificationPort interface.

    Calls module.configure() exactly once at construction (as required by
    GAP-6).  prompt_versions comes from module.runtime_metadata() if
    available.
    """

    def __init__(self, module: Any, **configure_kwargs: Any) -> None:
        self._module = module
        configure_fn = getattr(module, "configure", None)
        if callable(configure_fn):
            configure_fn(**configure_kwargs)
        meta_fn = getattr(module, "runtime_metadata", None)
        if callable(meta_fn):
            meta = meta_fn() or {}
            self._prompt_versions: dict = meta.get("prompt_versions", {}) or {}
        else:
            self._prompt_versions = {}

    @property
    def prompt_versions(self) -> dict:
        return self._prompt_versions

    def verify_draft(self, example: Any, draft: Any, evidence: list, llm: Any) -> Any:
        return self._module.verify_draft(example, draft, evidence, llm)

    def verify_claims(
        self,
        example: Any,
        draft: Any,
        claims: list,
        evidence_by_claim: dict,
        llm: Any,
    ) -> Any:
        return self._module.verify_claims(example, draft, claims, evidence_by_claim, llm)

    def revise_answer(self, example: Any, draft: Any, feedback: Any, llm: Any) -> Any:
        return self._module.revise_answer(example, draft, feedback, llm)


# ---------------------------------------------------------------------------
# AAnswerAdapter
# ---------------------------------------------------------------------------

class AAnswerAdapter:
    """Adapts A's AnswerAgent to E's DraftPort interface.

    Exposes generate_draft, self_reflect, independent_critique directly from
    the agent.  Also exposes run_m2() if A's baselines module provides it,
    and prompt_versions as a dict (from agent.prompt_versions).
    """

    def __init__(self, agent: Any) -> None:
        self._agent = agent

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

    def run_m2(self, example: Any, draft: Any, llm: Any) -> Any:
        """Call A's baselines.run_m2 if available; fall back to self_reflect."""
        try:
            from src.baselines import run_m2 as _baselines_run_m2  # lazy import
            return _baselines_run_m2(self._agent, example, draft, llm)
        except (ImportError, AttributeError):
            return self._agent.self_reflect(example, draft, llm)
