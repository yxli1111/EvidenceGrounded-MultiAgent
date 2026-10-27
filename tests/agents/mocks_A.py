"""Test-only mocks for Contributor A (parallel start, plan v2 section 4).

These are NOT substitute implementations of C's LLM client or D's revision: they are
scripted stand-ins that satisfy the contract Protocols so A's code can be tested before
integration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from contracts.schemas import (AnswerDraft, LLMResponse, LLMUsage, QAInput, RevisionResult,
                               Sentence, VerificationFeedback)

# --------------------------------------------------------------------------- mock QA examples
# Synthetic, HotpotQA-shaped examples (one bridge, one comparison). Gold is deliberately
# not part of QAInput; it is kept separately to mirror the QAInput/QAGold boundary.

BRIDGE_EXAMPLE = QAInput(
    example_id='mock-bridge-1',
    question='In which city was the director of the film "Silver Harbor" born?',
    context=[
        Sentence('Silver Harbor', 0, 'Silver Harbor is a 1998 drama film directed by Mara Lind.'),
        Sentence('Silver Harbor', 1, 'The film premiered at the Nordvik Film Festival.'),
        Sentence('Mara Lind', 0, 'Mara Lind (born 1961) is a Swedish film director.'),
        Sentence('Mara Lind', 1, 'Lind was born in Gothenburg and studied in Stockholm.'),
        Sentence('Nordvik', 0, 'Nordvik is a small coastal town.'),
    ],
    question_type='bridge',
)

COMPARISON_EXAMPLE = QAInput(
    example_id='mock-comparison-1',
    question='Are the Alder River and the Birch River located in the same country?',
    context=[
        Sentence('Alder River', 0, 'The Alder River is a river in northern Norway.'),
        Sentence('Birch River', 0, 'The Birch River flows through central Finland.'),
        Sentence('Birch River', 1, 'It is 120 km long.'),
    ],
    question_type='comparison',
)

MOCK_EXAMPLES = [BRIDGE_EXAMPLE, COMPARISON_EXAMPLE]
MOCK_GOLD = {'mock-bridge-1': 'Gothenburg', 'mock-comparison-1': 'no'}


def j(obj) -> str:
    return json.dumps(obj)


# --------------------------------------------------------------------------- mock LLM client

@dataclass
class Call:
    stage: str
    messages: list[dict[str, str]]
    model: str
    temperature: float
    max_tokens: int


@dataclass
class MockLLMClient:
    """Deterministic LLMClient: replies are chosen by stage.

    ``script`` maps stage -> reply text, list of reply texts (consumed in order), or a
    callable(Call) -> str. Stage names not in the script raise KeyError, so tests catch
    unexpected calls.
    """
    script: dict[str, str | list[str] | Callable[[Call], str]]
    calls: list[Call] = field(default_factory=list)

    def generate(self, *, stage: str, messages: list[dict[str, str]],
                 model: str, temperature: float, max_tokens: int) -> LLMResponse:
        call = Call(stage, [dict(m) for m in messages], model, temperature, max_tokens)
        self.calls.append(call)
        entry = self.script[stage]
        if callable(entry):
            text = entry(call)
        elif isinstance(entry, list):
            text = entry.pop(0)
        else:
            text = entry
        return LLMResponse(text=text, usage=LLMUsage(stage=stage, model=model,
                                                     input_tokens=None, output_tokens=None))

    def stages(self) -> list[str]:
        return [c.stage for c in self.calls]


# --------------------------------------------------------------------------- mock revision port

@dataclass
class MockRevisionPort:
    """Stands in for D's VerificationPort.revise_answer; records every call."""
    calls: list[tuple[QAInput, AnswerDraft, VerificationFeedback]] = field(default_factory=list)

    def verify_draft(self, *a, **k):  # not used by A
        raise NotImplementedError

    def verify_claims(self, *a, **k):  # not used by A
        raise NotImplementedError

    def revise_answer(self, example: QAInput, draft: AnswerDraft,
                      feedback: VerificationFeedback, llm) -> RevisionResult:
        self.calls.append((example, draft, feedback))
        return RevisionResult(draft_id=draft.draft_id, feedback_id=feedback.feedback_id,
                              final_answer=draft.short_answer + ' (revised)',
                              final_explanation=draft.explanation)
