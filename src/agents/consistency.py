"""Experiment A (Phase 2): answer-explanation consistency check (owner: Contributor A).

Research question: can detecting a mismatch between ``short_answer`` and
``explanation`` improve final-answer accuracy without introducing harmful revisions?

The checker reads ONLY the question, the draft's short answer and its explanation
(no context, no retrieved evidence, never gold). It returns consistent/inconsistent,
a rationale, and optionally a proposed short answer copied from the explanation.

Safety gate (enforced in code, not trusted to the prompt). The short answer is edited
only when ALL of the following hold; otherwise the draft is kept and the reason is logged:
  1. the checker says ``consistent == false`` and proposes a short answer;
  2. the proposal differs from the draft answer after normalisation;
  3. the proposal is short (<= MAX_PROPOSAL_WORDS words);
  4. the quoted explanation sentence occurs verbatim in the explanation, and
  5. the proposal occurs verbatim inside that quote (yes/no questions: the proposal is
     "yes"/"no" and the draft answer is the other one).
The explanation and the original draft are never modified. One LLM call (+ at most
``max_parse_repairs`` format repairs); no revision loop.
"""
from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass
from typing import Any

from contracts.interfaces import LLMClient
from contracts.schemas import (AnswerDraft, FeedbackItem, QAInput, RevisionResult,
                               VerificationFeedback)

from .answer import (AgentConfig, _as_text, as_draft, as_qa_input, call_and_parse,
                     load_prompt)

STAGE_CONSISTENCY = 'A.M1C.consistency_check'
METHOD_ID = 'M1C'
MAX_PROPOSAL_WORDS = 8

# gate outcomes (logged in ConsistencyDecision.gate)
GATE_CONSISTENT = 'consistent'            # checker found no mismatch
GATE_EDITED = 'edited'                    # mismatch flagged and short answer edited
GATE_NO_PROPOSAL = 'flag_no_proposal'     # flagged, but proposed nothing usable
GATE_SAME_ANSWER = 'flag_same_answer'     # proposal equals the draft answer
GATE_TOO_LONG = 'flag_too_long'           # proposal is not a short span
GATE_QUOTE_NOT_FOUND = 'flag_quote_not_in_explanation'
GATE_SPAN_NOT_IN_QUOTE = 'flag_span_not_in_quote'
GATE_YESNO_INVALID = 'flag_yesno_invalid'
GATE_BAD_TYPE = 'flag_unknown_mismatch_type'

# checker v2 must name one of these; v1 output has no type (None is accepted for it)
MISMATCH_TYPES = ('different_final_answer', 'stopped_at_intermediate_step', 'yes_no_reversed')


@dataclass
class ConsistencyDecision:
    """Everything needed to audit one check (stored with the record)."""
    flagged: bool                 # checker said inconsistent
    mismatch_type: str | None     # v2: one of MISMATCH_TYPES; v1: None
    edited: bool                  # gate passed and short_answer was changed
    gate: str                     # one of the GATE_* codes
    rationale: str
    proposed_short_answer: str | None
    explanation_quote: str | None
    original_short_answer: str
    final_short_answer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- normalisation

_ARTICLES = re.compile(r'\b(a|an|the)\b')
_WS = re.compile(r'\s+')


def _surface(text: str) -> str:
    """Lower-case, unify quotes/dashes, collapse whitespace (for verbatim containment)."""
    t = (text or '').lower()
    t = t.translate(str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"',
                                   '–': '-', '—': '-'}))
    return _WS.sub(' ', t).strip()


_EDGE = ' \'"`.,;:'   # quotes and sentence punctuation only; keeps ! ? ) that belong to names


def _strip_edges(text: str) -> str:
    return _surface(text).strip(_EDGE)


def _answer_key(text: str) -> str:
    """Answer comparison key: lower, no punctuation, no articles, collapsed spaces."""
    t = _surface(text)
    t = ''.join(ch for ch in t if ch not in string.punctuation)
    return _WS.sub(' ', _ARTICLES.sub(' ', t)).strip()


def _is_yes_no(text: str) -> bool:
    return _answer_key(text) in {'yes', 'no'}


# --------------------------------------------------------------------------- parsing

def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {'true', 'false'}:
        return value.strip().lower() == 'true'
    raise ValueError('"consistent" must be true or false')


def _opt_text(value: Any) -> str | None:
    t = _as_text(value)
    return None if not t or t.lower() in {'null', 'none', 'n/a'} else t


def parse_consistency_payload(obj: dict[str, Any]) -> tuple[bool, str, str | None, str | None, str | None]:
    if 'consistent' not in obj:
        raise ValueError('missing "consistent"')
    consistent = _as_bool(obj['consistent'])
    rationale = _as_text(obj.get('rationale', ''))
    proposed = _opt_text(obj.get('proposed_short_answer'))
    quote = _opt_text(obj.get('explanation_quote'))
    mtype = _opt_text(obj.get('mismatch_type'))
    return consistent, rationale, proposed, quote, mtype


# --------------------------------------------------------------------------- gate

def apply_gate(draft: AnswerDraft, consistent: bool, proposed: str | None,
               quote: str | None, mismatch_type: str | None = None) -> tuple[str, str | None]:
    """Return (gate_code, new_short_answer or None). Pure function, no LLM."""
    if consistent:
        return GATE_CONSISTENT, None
    if mismatch_type is not None and mismatch_type not in MISMATCH_TYPES:
        return GATE_BAD_TYPE, None
    if not proposed:
        return GATE_NO_PROPOSAL, None
    proposed = proposed.strip().strip('"').strip()
    if _answer_key(proposed) == _answer_key(draft.short_answer) or not _answer_key(proposed):
        return GATE_SAME_ANSWER, None
    expl = _surface(draft.explanation)
    if not quote or _strip_edges(quote) not in expl:
        return GATE_QUOTE_NOT_FOUND, None
    yn = _is_yes_no(draft.short_answer) or _is_yes_no(proposed)
    if yn or mismatch_type == 'yes_no_reversed':
        if not (_is_yes_no(draft.short_answer) and _is_yes_no(proposed)) or \
                mismatch_type not in (None, 'yes_no_reversed'):
            return GATE_YESNO_INVALID, None
        return GATE_EDITED, _answer_key(proposed)
    if len(proposed.split()) > MAX_PROPOSAL_WORDS:
        return GATE_TOO_LONG, None
    if _strip_edges(proposed) not in _surface(quote):
        return GATE_SPAN_NOT_IN_QUOTE, None
    return GATE_EDITED, _strip_edges_keep_case(proposed)


def _strip_edges_keep_case(text: str) -> str:
    return _WS.sub(' ', text).strip().strip(_EDGE + '\u2018\u2019\u201c\u201d')


# --------------------------------------------------------------------------- check

def consistency_feedback_id(draft: AnswerDraft) -> str:
    return f'{draft.draft_id}:{METHOD_ID}:consistency'


def check_consistency(example: QAInput, draft: AnswerDraft, llm: LLMClient,
                      config: AgentConfig) -> tuple[VerificationFeedback, RevisionResult | None,
                                                    ConsistencyDecision]:
    """One consistency check of the shared draft.

    Returns (feedback, revision, decision). ``revision`` is None unless the gate edited
    the short answer; the explanation is always the draft's own, unchanged.
    """
    example, draft = as_qa_input(example), as_draft(draft)
    prompt = load_prompt('answer.consistency_check')
    messages = prompt.render(question=example.question.strip(),
                             short_answer=draft.short_answer,
                             explanation=draft.explanation or '(empty)')
    consistent, rationale, proposed, quote, mtype = call_and_parse(
        llm, stage=STAGE_CONSISTENCY, messages=messages, config=config,
        parser=parse_consistency_payload)
    gate, new_answer = apply_gate(draft, consistent, proposed, quote, mtype)
    edited = gate == GATE_EDITED
    decision = ConsistencyDecision(
        flagged=not consistent, mismatch_type=None if consistent else mtype, edited=edited,
        gate=gate, rationale=rationale,
        proposed_short_answer=proposed, explanation_quote=quote,
        original_short_answer=draft.short_answer,
        final_short_answer=new_answer if edited else draft.short_answer)
    items = [] if consistent else [FeedbackItem(
        scope='draft', target_id=draft.draft_id, rationale=rationale or 'answer-explanation mismatch',
        label=None, evidence_ids=[], suggested_correction=proposed)]
    feedback = VerificationFeedback(feedback_id=consistency_feedback_id(draft), items=items,
                                    overall_comment=f'gate={gate}')
    revision = None
    if edited:
        revision = RevisionResult(draft_id=draft.draft_id, feedback_id=feedback.feedback_id,
                                  final_answer=new_answer, final_explanation=draft.explanation)
    return feedback, revision, decision
