"""Explicit test doubles. Never imported by contributor D's production code."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


@dataclass
class Sentence:
    title: str
    sentence_id: int
    text: str


@dataclass
class QAInput:
    example_id: str
    question: str
    context: list[Sentence]
    question_type: str | None = None


@dataclass
class AnswerDraft:
    draft_id: str
    short_answer: str
    explanation: str


@dataclass
class AtomicClaim:
    claim_id: str
    claim_text: str
    source_span: tuple[int, int] | None = None


@dataclass
class Evidence:
    doc_title: str
    sentence_id: int
    text: str
    score: float
    retrieval_query: str
    rank: int


@dataclass
class FeedbackItem:
    scope: str
    target_id: str
    rationale: str
    label: str | None
    evidence_ids: list[str] = field(default_factory=list)
    suggested_correction: str | None = None


@dataclass
class VerificationFeedback:
    feedback_id: str
    items: list[FeedbackItem]
    overall_comment: str


@dataclass
class RevisionResult:
    draft_id: str
    feedback_id: str
    final_answer: str
    final_explanation: str


@contextmanager
def mock_contracts():
    """Temporarily expose explicit mock schemas under the team's import path.

    This context is entered by offline tests/demo only. It does not install,
    create, or replace any real contracts files on disk.
    """
    package = ModuleType("contracts")
    package.__path__ = []
    schemas = ModuleType("contracts.schemas")
    for name in (
        "Sentence", "QAInput", "AnswerDraft", "AtomicClaim", "Evidence",
        "FeedbackItem", "VerificationFeedback", "RevisionResult",
    ):
        setattr(schemas, name, globals()[name])
    package.schemas = schemas
    with patch.dict(sys.modules, {"contracts": package, "contracts.schemas": schemas}):
        yield schemas


class ScriptedLLM:
    """Record calls and return predetermined responses; never connects online."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("Unexpected LLM call: no scripted response remains")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            response = json.dumps(response, ensure_ascii=False, allow_nan=False)
        if not isinstance(response, str):
            raise TypeError("Scripted responses must be dict, str or Exception")
        return SimpleNamespace(text=response)


def sample_inputs():
    """A small multi-sentence example; no benchmark or gold labels are used."""
    context = [
        Sentence("Ada Lovelace", 0, "Ada Lovelace was born in London."),
        Sentence("London", 0, "London is in England."),
        Sentence("Ada Lovelace", 1, "Ada Lovelace was born on 10 December 1815."),
    ]
    example = QAInput(
        "example-1", "In what country was Ada Lovelace born?", context, "bridge"
    )
    draft = AnswerDraft(
        "example-1:draft", "France", "Ada Lovelace was born in Paris, France."
    )
    claims = [AtomicClaim("claim-1", "Ada Lovelace was born in France.")]
    evidence = [
        Evidence(sentence.title, sentence.sentence_id, sentence.text,
                 0.9 - index * 0.1, example.question, index + 1)
        for index, sentence in enumerate(context[:2])
    ]
    return example, draft, claims, evidence
