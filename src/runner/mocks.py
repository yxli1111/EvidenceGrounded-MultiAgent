"""Mock dataclasses and port implementations for offline testing.

Nothing here should be used in production.  Real contracts/schemas.py and
the real A/B/C/D implementations replace everything in this file once they
exist.  Tests import directly from this module; no production path touches it.

Mock dataclasses deliberately mirror the field names that contracts/schemas.py
will eventually define (see evaluation/E_CONTRACT_PROPOSAL.md GAP-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Mock schema dataclasses (stand-ins for contracts/schemas.py)
# ---------------------------------------------------------------------------

@dataclass
class MockSentence:
    title: str
    sentence_id: int
    text: str


@dataclass
class MockQAInput:
    example_id: str
    question: str
    context: list[MockSentence]
    question_type: str = "factoid"


@dataclass
class MockQAGold:
    example_id: str
    gold_answer: str
    gold_supporting_facts: list[dict]   # [{"title": str, "sentence_id": int}]


@dataclass
class MockAnswerDraft:
    draft_id: str
    short_answer: str
    explanation: str


@dataclass
class MockAtomicClaim:
    claim_id: str
    claim_text: str
    source_span: tuple[int, int] | None = None


@dataclass
class MockEvidence:
    doc_title: str
    sentence_id: int
    text: str
    score: float
    retrieval_query: str
    rank: int


@dataclass
class MockFeedbackItem:
    scope: str                          # "draft" | "claim"
    target_id: str
    rationale: str
    label: str | None                   # "SUPPORTED" | "CONTRADICTED" | "UNSUPPORTED" | None
    evidence_ids: list[str]
    suggested_correction: str | None = None


@dataclass
class MockVerificationFeedback:
    feedback_id: str
    items: list[MockFeedbackItem]
    overall_comment: str


@dataclass
class MockRevisionResult:
    draft_id: str
    feedback_id: str
    final_answer: str
    final_explanation: str


@dataclass
class MockLLMUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: float | None = None
    cache_hit: bool | None = None


@dataclass
class MockLLMResponse:
    text: str
    usage: MockLLMUsage | None = None
    stage: str = ""
    model: str = ""


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Canned-response LLM client with usage accumulation.

    Call drain_usage() after each method run to collect and clear the log.
    The runner calls this automatically via getattr duck-typing.
    """

    def __init__(self, model: str = "mock-model-v1") -> None:
        self.model = model
        self._usage_log: list[dict] = []

    def generate(
        self,
        stage: str,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> MockLLMResponse:
        usage_dict = {
            "stage": stage,
            "model": model,
            "input_tokens": 50,
            "output_tokens": 80,
            "cost_usd": 0.00013,
            "latency_ms": 120.0,
            "cache_hit": False,
        }
        self._usage_log.append(usage_dict)
        return MockLLMResponse(
            text='{"mock": "response"}',
            usage=MockLLMUsage(
                input_tokens=50,
                output_tokens=80,
                cost_usd=0.00013,
                latency_ms=120.0,
                cache_hit=False,
            ),
            stage=stage,
            model=model,
        )

    def drain_usage(self) -> list[dict]:
        """Return accumulated usage records and clear the internal log."""
        log = list(self._usage_log)
        self._usage_log.clear()
        return log


# ---------------------------------------------------------------------------
# Mock port implementations
# ---------------------------------------------------------------------------

class MockDraftPort:
    """A stub for A's draft and critique functions."""

    def __init__(self) -> None:
        self.generate_draft_call_count = 0

    def generate_draft(self, example: Any, llm: Any) -> MockAnswerDraft:
        self.generate_draft_call_count += 1
        eid = _attr(example, "example_id")
        llm.generate("generate_draft", [], _attr(llm, "model", "mock"), 0.0, 512)
        return MockAnswerDraft(
            draft_id=f"{eid}:draft",
            short_answer="The mock answer.",
            explanation="This is a mock explanation of the answer.",
        )

    def self_reflect(self, example: Any, draft: Any, llm: Any) -> MockRevisionResult:
        llm.generate("self_reflect", [], _attr(llm, "model", "mock"), 0.0, 512)
        return MockRevisionResult(
            draft_id=_attr(draft, "draft_id"),
            feedback_id=f"{_attr(draft, 'draft_id')}:self-reflect",
            final_answer="The revised mock answer (M2).",
            final_explanation="Revised explanation after self-reflection (M2).",
        )

    def independent_critique(
        self, example: Any, draft: Any, llm: Any
    ) -> MockVerificationFeedback:
        did = _attr(draft, "draft_id")
        llm.generate("independent_critique", [], _attr(llm, "model", "mock"), 0.0, 1024)
        return MockVerificationFeedback(
            feedback_id=f"{did}:critique",
            items=[
                MockFeedbackItem(
                    scope="draft",
                    target_id=did,
                    rationale="Mock critique: the explanation could be more specific.",
                    label=None,         # M3 must not invent evidence verdicts
                    evidence_ids=[],
                    suggested_correction="Add a specific example to support the claim.",
                )
            ],
            overall_comment="Mock independent critique.",
        )


class MockRetrievePort:
    """Returns the first top_k context sentences as mock evidence."""

    def retrieve(self, query: str, context: Any, top_k: int = 5) -> list[MockEvidence]:
        sentences = list(context)[:top_k]
        results = []
        for rank, sent in enumerate(sentences, start=1):
            if isinstance(sent, dict):
                title, sid, text = sent["title"], sent["sentence_id"], sent["text"]
            else:
                title, sid, text = sent.title, sent.sentence_id, sent.text
            results.append(MockEvidence(
                doc_title=title,
                sentence_id=sid,
                text=text,
                score=round(1.0 - 0.05 * rank, 4),
                retrieval_query=query,
                rank=rank,
            ))
        return results


class MockClaimsPort:
    """Returns two canned atomic claims derived from the draft."""

    def extract_claims(self, draft: Any, llm: Any) -> list[MockAtomicClaim]:
        did = _attr(draft, "draft_id")
        llm.generate("extract_claims", [], _attr(llm, "model", "mock"), 0.0, 512)
        return [
            MockAtomicClaim(
                claim_id=f"{did}:claim_0",
                claim_text="The mock answer is factually correct.",
                source_span=None,
            ),
            MockAtomicClaim(
                claim_id=f"{did}:claim_1",
                claim_text="The explanation provides supporting context.",
                source_span=None,
            ),
        ]


class MockVerificationPort:
    """Mock implementations of D's verify_draft, verify_claims, revise_answer."""

    def verify_draft(
        self,
        example: Any,
        draft: Any,
        evidence: list[Any],
        llm: Any,
    ) -> MockVerificationFeedback:
        did = _attr(draft, "draft_id")
        llm.generate("verify_draft", [], _attr(llm, "model", "mock"), 0.0, 1024)
        ev_id = (
            f"{_attr(evidence[0], 'doc_title')}::{_attr(evidence[0], 'sentence_id')}"
            if evidence else ""
        )
        return MockVerificationFeedback(
            feedback_id=f"{did}:verify_draft",
            items=[
                MockFeedbackItem(
                    scope="draft",
                    target_id=did,
                    rationale="Mock verification: evidence supports the draft.",
                    label="SUPPORTED",
                    evidence_ids=[ev_id] if ev_id else [],
                    suggested_correction=None,
                )
            ],
            overall_comment="Mock whole-draft verification.",
        )

    def verify_claims(
        self,
        example: Any,
        draft: Any,
        claims: list[Any],
        evidence_by_claim: dict[str, list[Any]],
        llm: Any,
    ) -> MockVerificationFeedback:
        did = _attr(draft, "draft_id")
        llm.generate("verify_claims", [], _attr(llm, "model", "mock"), 0.0, 1024)
        items = []
        for claim in claims:
            cid = _attr(claim, "claim_id")
            evs = evidence_by_claim.get(cid, [])
            ev_id = (
                f"{_attr(evs[0], 'doc_title')}::{_attr(evs[0], 'sentence_id')}"
                if evs else ""
            )
            items.append(MockFeedbackItem(
                scope="claim",
                target_id=cid,
                rationale=f"Mock claim check for {cid}.",
                label="SUPPORTED" if evs else "UNSUPPORTED",
                evidence_ids=[ev_id] if ev_id else [],
                suggested_correction=None,
            ))
        return MockVerificationFeedback(
            feedback_id=f"{did}:verify_claims",
            items=items,
            overall_comment="Mock claim-level verification.",
        )

    def revise_answer(
        self,
        example: Any,
        draft: Any,
        feedback: Any,
        llm: Any,
    ) -> MockRevisionResult:
        did = _attr(draft, "draft_id")
        fid = _attr(feedback, "feedback_id")
        llm.generate("revise_answer", [], _attr(llm, "model", "mock"), 0.0, 512)
        return MockRevisionResult(
            draft_id=did,
            feedback_id=fid,
            final_answer="The revised mock answer.",
            final_explanation="Mock revised explanation incorporating feedback.",
        )


# ---------------------------------------------------------------------------
# Synthetic example factory
# ---------------------------------------------------------------------------

def make_synthetic_example(
    example_id: str = "synth_001",
    question: str = "What is the capital of France?",
) -> MockQAInput:
    return MockQAInput(
        example_id=example_id,
        question=question,
        context=[
            MockSentence("France", 0, "Paris is the capital of France."),
            MockSentence("France", 1, "France is a country in Western Europe."),
            MockSentence("Germany", 0, "Berlin is the capital of Germany."),
            MockSentence("Germany", 1, "Germany is located in Central Europe."),
            MockSentence("Europe", 0, "Europe is a continent with many countries."),
        ],
        question_type="factoid",
    )


def make_synthetic_gold(
    example_id: str = "synth_001",
    gold_answer: str = "Paris",
) -> MockQAGold:
    return MockQAGold(
        example_id=example_id,
        gold_answer=gold_answer,
        gold_supporting_facts=[{"title": "France", "sentence_id": 0}],
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
