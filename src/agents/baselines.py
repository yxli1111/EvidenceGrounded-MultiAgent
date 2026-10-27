"""Baseline conditions M1/M2 and M3 wiring helper (owner: Contributor A).

* **M1 Direct generation** -- the shared draft is the final output; no critique, no
  revision, and **no RevisionResult** (PROJECT_CONTRACT section 2).
* **M2 Self-reflection** -- the same Answer Agent checks its own draft
  (``self_check``) and then revises it exactly once (``self_revise``). Two LLM calls,
  mirroring M3's critic + revision. M2 deliberately uses A's own revision prompt; it is
  a distinct baseline by design.
* **M3 Independent critic** -- ``run_m3`` = A's ``independent_critique`` followed by
  exactly one call to D's shared ``revise_answer`` through the ``VerificationPort``
  contract. A never implements an M3 revision prompt.

E's runner is the orchestration owner; ``run_m1/run_m2/run_m3`` are thin, contract-only
helpers for A's own tests and research notebook, and E may call the port methods
directly instead. All helpers take the **shared, already-generated draft** so that the
same draft is reused across M1-M5.
"""
from __future__ import annotations

from dataclasses import dataclass

from contracts.interfaces import LLMClient, VerificationPort
from contracts.schemas import AnswerDraft, QAInput, RevisionResult, VerificationFeedback

from .answer import (STAGE_SELF_CHECK, STAGE_SELF_REVISE, AgentConfig, call_and_parse,
                     load_prompt, parse_answer_payload, parse_issues_payload,
                     render_context)
from .independent_critic import independent_critique, issues_to_feedback


def self_check_feedback_id(draft: AnswerDraft) -> str:
    return f'{draft.draft_id}:M2:self_reflection'


# --------------------------------------------------------------------------- M2

def self_check(example: QAInput, draft: AnswerDraft, llm: LLMClient,
               config: AgentConfig) -> VerificationFeedback:
    """M2 step 1: the Answer Agent reviews its own draft (feedback only)."""
    prompt = load_prompt('baselines.self_check')
    messages = prompt.render(question=example.question.strip(),
                             context=render_context(example.context),
                             short_answer=draft.short_answer,
                             explanation=draft.explanation or '(empty)')
    issues, overall = call_and_parse(llm, stage=STAGE_SELF_CHECK, messages=messages,
                                     config=config, parser=parse_issues_payload)
    return issues_to_feedback(self_check_feedback_id(draft), draft, issues, overall)


def format_review(feedback: VerificationFeedback) -> str:
    if not feedback.items:
        return feedback.overall_comment or 'No issues found.'
    lines = []
    for i, item in enumerate(feedback.items, 1):
        line = f'{i}. {item.rationale}'
        if item.suggested_correction:
            line += f' Suggested correction: {item.suggested_correction}'
        lines.append(line)
    if feedback.overall_comment:
        lines.append(f'Overall: {feedback.overall_comment}')
    return '\n'.join(lines)


def self_revise(example: QAInput, draft: AnswerDraft, feedback: VerificationFeedback,
                llm: LLMClient, config: AgentConfig) -> RevisionResult:
    """M2 step 2: exactly one self-revision conditioned on the self-check."""
    prompt = load_prompt('baselines.self_revise')
    messages = prompt.render(question=example.question.strip(),
                             context=render_context(example.context),
                             short_answer=draft.short_answer,
                             explanation=draft.explanation or '(empty)',
                             review=format_review(feedback))
    short, expl = call_and_parse(llm, stage=STAGE_SELF_REVISE, messages=messages,
                                 config=config, parser=parse_answer_payload)
    return RevisionResult(draft_id=draft.draft_id, feedback_id=feedback.feedback_id,
                          final_answer=short, final_explanation=expl)


def self_reflect(example: QAInput, draft: AnswerDraft, llm: LLMClient,
                 config: AgentConfig) -> RevisionResult:
    """AnswerPort.self_reflect: self-check then one self-revision."""
    return run_m2(example, draft, llm, config).final


# --------------------------------------------------------------------------- method runners

@dataclass
class BaselineOutput:
    """What A's helpers return; E maps this onto ExperimentRecord fields.

    For M1, ``final`` is None (the draft is the final output) and ``feedback`` is None.
    """
    method_id: str
    draft: AnswerDraft
    feedback: VerificationFeedback | None
    final: RevisionResult | None

    @property
    def final_answer(self) -> str:
        return self.final.final_answer if self.final else self.draft.short_answer

    @property
    def final_explanation(self) -> str:
        return self.final.final_explanation if self.final else self.draft.explanation


def run_m1(example: QAInput, draft: AnswerDraft) -> BaselineOutput:
    """M1: no LLM call beyond the shared draft."""
    return BaselineOutput('M1', draft, None, None)


def run_m2(example: QAInput, draft: AnswerDraft, llm: LLMClient,
           config: AgentConfig) -> BaselineOutput:
    feedback = self_check(example, draft, llm, config)
    final = self_revise(example, draft, feedback, llm, config)
    return BaselineOutput('M2', draft, feedback, final)


def run_m3(example: QAInput, draft: AnswerDraft, llm: LLMClient, config: AgentConfig,
           revision: VerificationPort) -> BaselineOutput:
    """M3: A's critic feedback, then D's shared ``revise_answer`` exactly once."""
    feedback = independent_critique(example, draft, llm, config)
    final = revision.revise_answer(example, draft, feedback, llm)
    if not isinstance(final, RevisionResult):
        raise TypeError('revise_answer must return contracts.schemas.RevisionResult')
    return BaselineOutput('M3', draft, feedback, final)
