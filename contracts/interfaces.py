"""Public callable ports; implementation owned by contributors A-E.
Use mocks in unit tests. Signatures are normative for v0.1 review.
"""
from __future__ import annotations
from typing import Protocol
from .schemas import (QAInput, QAGold, AnswerDraft, AtomicClaim, Evidence,
                      VerificationFeedback, RevisionResult, LLMResponse, LLMUsage)

class LLMClient(Protocol):
    def generate(self, *, stage: str, messages: list[dict[str, str]],
                 model: str, temperature: float, max_tokens: int) -> LLMResponse: ...

class DatasetPort(Protocol):
    def load_split(self, split_name: str) -> tuple[list[QAInput], dict[str, QAGold]]: ...

class AnswerPort(Protocol):
    def generate_draft(self, example: QAInput, llm: LLMClient) -> AnswerDraft: ...
    def self_reflect(self, example: QAInput, draft: AnswerDraft,
                     llm: LLMClient) -> RevisionResult: ...
    def independent_critique(self, example: QAInput, draft: AnswerDraft,
                             llm: LLMClient) -> VerificationFeedback: ...

class ClaimPort(Protocol):
    def extract_claims(self, draft: AnswerDraft, llm: LLMClient) -> list[AtomicClaim]: ...

class RetrievalPort(Protocol):
    def retrieve(self, query: str, context: list, top_k: int = 5) -> list[Evidence]: ...

class VerificationPort(Protocol):
    def verify_draft(self, example: QAInput, draft: AnswerDraft,
                     evidence: list[Evidence], llm: LLMClient) -> VerificationFeedback: ...
    def verify_claims(self, example: QAInput, draft: AnswerDraft,
                      claims: list[AtomicClaim],
                      evidence_by_claim: dict[str, list[Evidence]],
                      llm: LLMClient) -> VerificationFeedback: ...
    def revise_answer(self, example: QAInput, draft: AnswerDraft,
                      feedback: VerificationFeedback, llm: LLMClient) -> RevisionResult: ...
