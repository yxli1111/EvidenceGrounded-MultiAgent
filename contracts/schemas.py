"""Shared v0.1 wire schemas. Python 3.11, standard library only.
JSON serialization: dataclasses.asdict; deserialization: explicit constructors/adapters.
Do not change without team approval.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

MethodID = Literal['M1', 'M2', 'M3', 'M4', 'M5']
Verdict = Literal['SUPPORTED', 'CONTRADICTED', 'UNSUPPORTED']
Scope = Literal['draft', 'claim']
Status = Literal['ok', 'error', 'skipped']

@dataclass
class Sentence:
    title: str
    sentence_id: int  # 0-based index within original HotpotQA title
    text: str

@dataclass
class QAInput:
    example_id: str
    question: str
    context: list[Sentence]
    question_type: str | None = None

@dataclass
class QAGold:
    example_id: str
    gold_answer: str
    gold_supporting_facts: list[tuple[str, int]]

@dataclass
class AnswerDraft:
    draft_id: str
    short_answer: str
    explanation: str

@dataclass
class AtomicClaim:
    claim_id: str
    claim_text: str
    source_span: tuple[int, int] | None = None  # [start,end) offsets in explanation

@dataclass
class Evidence:
    doc_title: str
    sentence_id: int
    text: str
    score: float
    retrieval_query: str
    rank: int  # 1-based
    @property
    def evidence_id(self) -> str:
        return f'{self.doc_title}::{self.sentence_id}'

@dataclass
class FeedbackItem:
    scope: Scope
    target_id: str  # draft_id for draft, claim_id for claim
    rationale: str
    label: Verdict | None = None  # M3 critic may not make evidence verdict
    evidence_ids: list[str] = field(default_factory=list)
    suggested_correction: str | None = None

@dataclass
class VerificationFeedback:
    feedback_id: str
    items: list[FeedbackItem]
    overall_comment: str = ''

@dataclass
class RevisionResult:
    draft_id: str
    feedback_id: str
    final_answer: str
    final_explanation: str

@dataclass
class LLMUsage:
    stage: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_seconds: float | None = None
    cache_hit: bool = False

@dataclass
class LLMResponse:
    text: str
    usage: LLMUsage

@dataclass
class ExperimentRecord:
    example_id: str
    method_id: MethodID
    status: Status
    draft: AnswerDraft | None = None
    final: RevisionResult | None = None  # M1: null; final output is draft
    claims: list[AtomicClaim] = field(default_factory=list)
    evidence: dict[str, list[Evidence]] = field(default_factory=dict)  # query/claim ID -> hits
    feedback: VerificationFeedback | None = None
    usage: list[LLMUsage] = field(default_factory=list)
    model_config: dict = field(default_factory=dict)
    prompt_versions: dict[str, str] = field(default_factory=dict)
    error: str | None = None
