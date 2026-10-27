"""M3 Independent Critic (owner: Contributor A).

A second LLM agent reviews the shared draft using the question and the same
benchmark-provided context, **without separately retrieved evidence**. It returns
feedback only, in the common ``VerificationFeedback`` schema; revision is done by
D's shared ``revise_answer`` (called by E's runner, or by ``baselines.run_m3``).

Contract rules enforced here (PROJECT_CONTRACT section 2):
* every item has ``scope='draft'`` and ``target_id=draft.draft_id``;
* ``label is None`` and ``evidence_ids == []`` -- the critic must not invent evidence
  verdicts or citations, even if the model emits them;
* empty ``items`` means "no actionable criticism".
"""
from __future__ import annotations

from contracts.interfaces import LLMClient
from contracts.schemas import AnswerDraft, FeedbackItem, QAInput, VerificationFeedback

from .answer import (STAGE_CRITIC, AgentConfig, call_and_parse, load_prompt,
                     parse_issues_payload, render_context)


def critic_feedback_id(draft: AnswerDraft) -> str:
    return f'{draft.draft_id}:M3:critic'


def issues_to_feedback(feedback_id: str, draft: AnswerDraft,
                       issues: list[tuple[str, str | None]],
                       overall_comment: str) -> VerificationFeedback:
    """Map parsed critic issues to contract feedback (no labels, no evidence ids)."""
    items = [FeedbackItem(scope='draft', target_id=draft.draft_id, rationale=rationale,
                          label=None, evidence_ids=[], suggested_correction=correction)
             for rationale, correction in issues]
    return VerificationFeedback(feedback_id=feedback_id, items=items,
                                overall_comment=overall_comment)


def independent_critique(example: QAInput, draft: AnswerDraft, llm: LLMClient,
                         config: AgentConfig) -> VerificationFeedback:
    """One critic call (+ optional format repair). Returns feedback only, never a revision."""
    prompt = load_prompt('baselines.independent_critic')
    messages = prompt.render(question=example.question.strip(),
                             context=render_context(example.context),
                             short_answer=draft.short_answer,
                             explanation=draft.explanation or '(empty)')
    issues, overall = call_and_parse(llm, stage=STAGE_CRITIC, messages=messages,
                                     config=config, parser=parse_issues_payload)
    return issues_to_feedback(critic_feedback_id(draft), draft, issues, overall)
