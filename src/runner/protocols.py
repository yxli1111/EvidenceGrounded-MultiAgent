"""E-owned Protocol interfaces for every external port used by the runner.

These are the only contracts E is allowed to define.  Real implementations
(A, B, C, D) must satisfy these signatures when wired in.  Until then, all
ports are backed by mocks in src/runner/mocks.py.

Dataclass types from contracts/schemas.py are referenced as Any because
that package does not yet exist.  When it is created, static type checkers
can be tightened by replacing Any with the concrete types.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """C's LLM client — all LLM calls go through this interface."""

    def generate(
        self,
        *,
        stage: str,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        """Return an LLMResponse with at least a .text attribute and optional .usage."""
        ...


@runtime_checkable
class DraftPort(Protocol):
    """A's answer-generation and critique interface."""

    def generate_draft(self, example: Any, llm: LLMClient) -> Any:
        """Return AnswerDraft: draft_id, short_answer, explanation."""
        ...

    def self_reflect(self, example: Any, draft: Any, llm: LLMClient) -> Any:
        """M2: return RevisionResult directly (no external verifier)."""
        ...

    def independent_critique(self, example: Any, draft: Any, llm: LLMClient) -> Any:
        """M3: return VerificationFeedback with label=None, evidence_ids=[] per item."""
        ...


@runtime_checkable
class RetrievePort(Protocol):
    """B's evidence retrieval interface."""

    def retrieve(self, query: str, context: Any, top_k: int) -> list[Any]:
        """Return list of Evidence objects ranked by relevance (top_k max)."""
        ...


@runtime_checkable
class ClaimsPort(Protocol):
    """C's atomic-claim extraction interface."""

    def extract_claims(self, draft: Any, llm: LLMClient) -> list[Any]:
        """Return list[AtomicClaim]; empty list is valid (no factual claims)."""
        ...


@runtime_checkable
class VerificationPort(Protocol):
    """D's verification and shared revision interface (M3/M4/M5)."""

    def verify_draft(
        self,
        example: Any,
        draft: Any,
        evidence: list[Any],
        llm: LLMClient,
    ) -> Any:
        """M4: whole-draft verdict. Return VerificationFeedback."""
        ...

    def verify_claims(
        self,
        example: Any,
        draft: Any,
        claims: list[Any],
        evidence_by_claim: dict[str, list[Any]],
        llm: LLMClient,
    ) -> Any:
        """M5: per-claim verdict. evidence_by_claim must include every claim_id."""
        ...

    def revise_answer(
        self,
        example: Any,
        draft: Any,
        feedback: Any,
        llm: LLMClient,
    ) -> Any:
        """M3/M4/M5 shared revision. Return RevisionResult. Call at most once."""
        ...
