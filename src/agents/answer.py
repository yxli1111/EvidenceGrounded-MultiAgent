"""Answer Agent: shared initial draft generation (owner: Contributor A).

Public API
----------
- ``AgentConfig``            generation settings A passes to ``LLMClient.generate``.
- ``generate_draft``         one structured draft per example (shared by M1-M5).
- ``AnswerAgent``            thin adapter implementing ``contracts.interfaces.AnswerPort``.
- ``AgentOutputError``       raised when the LLM output cannot be parsed after repair.
- ``PROMPT_VERSIONS``        prompt id -> version string (name@sha256[:10]) for logging.

Helpers shared by ``baselines.py`` and ``independent_critic.py`` (prompt loading,
context rendering, JSON parsing) also live here so that A owns exactly one copy.

Design notes
------------
* QAGold is never accepted or used here: every function takes ``QAInput`` only.
* ``draft_id`` is reproducible: ``f"{example_id}:draft"`` (PROJECT_CONTRACT section 2).
* All LLM calls go through C's ``LLMClient``; usage/latency is recorded there / by E's
  runner. A does not retry transport errors (C does). A performs at most
  ``AgentConfig.max_parse_repairs`` *format-repair* calls when the reply is not
  valid JSON. The repair request differs from the original request (it contains the
  bad reply), so it never collides with C's exact-request cache.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any

from contracts.interfaces import LLMClient
from contracts.schemas import (AnswerDraft, QAInput, RevisionResult, Sentence,
                               VerificationFeedback)

# --------------------------------------------------------------------------- config

@dataclass(frozen=True)
class AgentConfig:
    """Generation settings for A's LLM calls.

    The concrete values (model, temperature, max_tokens) are team-owned and should
    be read from ``configs/experiment.yaml`` by E's runner and passed in here. There
    is deliberately no default model, so an unset backbone fails loudly.
    """
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    max_parse_repairs: int = 1  # extra calls allowed when the reply is not valid JSON

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> 'AgentConfig':
        known = {k: d[k] for k in ('model', 'temperature', 'max_tokens', 'max_parse_repairs') if k in d}
        return cls(**known)

    def as_log_dict(self) -> dict[str, Any]:
        return {'model': self.model, 'temperature': self.temperature,
                'max_tokens': self.max_tokens, 'max_parse_repairs': self.max_parse_repairs}


class AgentOutputError(ValueError):
    """LLM output could not be parsed into the required structure.

    ``raw_outputs`` keeps every raw reply so E can store them in the
    ExperimentRecord (status='error') for auditing.
    """
    def __init__(self, message: str, raw_outputs: list[str] | None = None):
        super().__init__(message)
        self.raw_outputs = list(raw_outputs or [])


# --------------------------------------------------------------------------- prompts

PROMPTS_DIR = Path(__file__).resolve().parents[2] / 'prompts'

# prompt id -> relative file path (under prompts/)
PROMPT_FILES: dict[str, str] = {
    'answer.draft': 'answer/draft_v1.md',
    'baselines.self_check': 'baselines/self_check_v1.md',
    'baselines.self_revise': 'baselines/self_revise_v1.md',
    'baselines.independent_critic': 'baselines/independent_critic_v1.md',
}

_SECTION_RE = re.compile(r'^### (SYSTEM|USER)\s*$', re.MULTILINE)


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    version: str          # e.g. 'draft_v1@1a2b3c4d5e'
    system: Template
    user: Template

    def render(self, **values: str) -> list[dict[str, str]]:
        return [
            {'role': 'system', 'content': self.system.substitute(values).strip()},
            {'role': 'user', 'content': self.user.substitute(values).strip()},
        ]


def load_prompt(prompt_id: str) -> PromptTemplate:
    rel = PROMPT_FILES[prompt_id]
    # Normalize line endings so Windows (CRLF) and macOS/Linux checkouts give the same
    # rendered prompt and the same version hash.
    raw = (PROMPTS_DIR / rel).read_text(encoding='utf-8').replace('\r\n', '\n')
    parts = _SECTION_RE.split(raw)
    # parts = [preamble, 'SYSTEM', sys_text, 'USER', user_text]
    sections = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
    if set(sections) != {'SYSTEM', 'USER'}:
        raise ValueError(f'prompt {rel} must contain exactly one ### SYSTEM and one ### USER section')
    digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:10]
    return PromptTemplate(prompt_id=prompt_id, version=f'{Path(rel).stem}@{digest}',
                          system=Template(sections['SYSTEM']), user=Template(sections['USER']))


def prompt_versions() -> dict[str, str]:
    """Prompt id -> version string, for ExperimentRecord.prompt_versions."""
    return {pid: load_prompt(pid).version for pid in PROMPT_FILES}


PROMPT_VERSIONS: dict[str, str] = prompt_versions()

# Stage names passed to LLMClient.generate (used by C/E for usage accounting).
STAGE_DRAFT = 'A.draft'
STAGE_SELF_CHECK = 'A.M2.self_check'
STAGE_SELF_REVISE = 'A.M2.self_revise'
STAGE_CRITIC = 'A.M3.independent_critic'


# --------------------------------------------------------------------------- rendering

def render_context(context: list[Sentence]) -> str:
    """Render HotpotQA sentences grouped by title, keeping original (title, sentence_id).

    Order of titles and sentences is preserved exactly as given by B's loader.
    """
    blocks: list[str] = []
    current: str | None = None
    lines: list[str] = []
    for s in context:
        if s.title != current:
            if current is not None:
                blocks.append('\n'.join(lines))
            current, lines = s.title, [f'[{s.title}]']
        lines.append(f'  ({s.sentence_id}) {s.text.strip()}')
    if current is not None:
        blocks.append('\n'.join(lines))
    return '\n\n'.join(blocks) if blocks else '(no context provided)'


def draft_id_for(example: QAInput) -> str:
    return f'{example.example_id}:draft'


# --------------------------------------------------------------------------- JSON parsing

_FENCE_RE = re.compile(r'^\s*```(?:json)?\s*(.*?)\s*```\s*$', re.DOTALL | re.IGNORECASE)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object from an LLM reply (tolerates ``` fences and surrounding prose)."""
    candidate = text.strip()
    m = _FENCE_RE.match(candidate)
    if m:
        candidate = m.group(1)
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find('{'), candidate.rfind('}')
        if start == -1 or end <= start:
            raise ValueError('no JSON object found in reply')
        obj = json.loads(candidate[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError('reply JSON is not an object')
    return obj


def _as_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    raise ValueError(f'expected a string, got {type(value).__name__}')


def parse_answer_payload(obj: dict[str, Any]) -> tuple[str, str]:
    """Validate {"short_answer", "explanation"}; returns (short_answer, explanation)."""
    if 'short_answer' not in obj:
        raise ValueError('missing "short_answer"')
    short = _as_text(obj['short_answer'])
    if not short:
        raise ValueError('"short_answer" is empty')
    expl = _as_text(obj.get('explanation', ''))
    return short, expl


def parse_issues_payload(obj: dict[str, Any]) -> tuple[list[tuple[str, str | None]], str]:
    """Validate {"issues": [{"rationale", "suggested_correction"}], "overall_comment"}."""
    issues_raw = obj.get('issues', [])
    if issues_raw is None:
        issues_raw = []
    if not isinstance(issues_raw, list):
        raise ValueError('"issues" must be a list')
    issues: list[tuple[str, str | None]] = []
    for it in issues_raw:
        if isinstance(it, str):
            rationale, corr = it.strip(), None
        elif isinstance(it, dict):
            rationale = _as_text(it.get('rationale') or it.get('problem') or it.get('description'))
            corr_raw = it.get('suggested_correction')
            corr = _as_text(corr_raw) if corr_raw is not None else None
            if corr is not None and (corr == '' or corr.lower() in {'null', 'none', 'n/a'}):
                corr = None
        else:
            raise ValueError('each issue must be an object or string')
        if rationale:
            issues.append((rationale, corr))
    overall = _as_text(obj.get('overall_comment', ''))
    return issues, overall


def call_and_parse(llm: LLMClient, *, stage: str, messages: list[dict[str, str]],
                   config: AgentConfig, parser) -> Any:
    """Call the LLM and parse the reply; on a format failure, issue bounded repair calls."""
    raw_outputs: list[str] = []
    msgs = list(messages)
    last_err: Exception | None = None
    for attempt in range(config.max_parse_repairs + 1):
        resp = llm.generate(stage=stage if attempt == 0 else f'{stage}.repair',
                            messages=msgs, model=config.model,
                            temperature=config.temperature, max_tokens=config.max_tokens)
        raw_outputs.append(resp.text)
        try:
            return parser(parse_json_object(resp.text))
        except (ValueError, json.JSONDecodeError) as err:
            last_err = err
            msgs = list(messages) + [
                {'role': 'assistant', 'content': resp.text},
                {'role': 'user', 'content': (
                    f'Your previous reply could not be used ({err}). Reply again with only the '
                    'JSON object in exactly the requested format.')},
            ]
    raise AgentOutputError(f'{stage}: unparseable LLM output after '
                           f'{config.max_parse_repairs + 1} attempt(s): {last_err}', raw_outputs)


# --------------------------------------------------------------------------- input adapters

def as_qa_input(example: Any) -> QAInput:
    """Return a contract ``QAInput``; also accepts the equivalent plain dict.

    B's loader currently yields dicts, so A accepts both. This only changes the
    container type. Field names and values must already follow the contract.
    """
    if isinstance(example, QAInput):
        return example
    if not isinstance(example, dict):
        raise TypeError(f'expected QAInput or dict, got {type(example).__name__}')
    ctx = [s if isinstance(s, Sentence) else
           Sentence(title=s['title'], sentence_id=int(s['sentence_id']), text=s['text'])
           for s in example['context']]
    return QAInput(example_id=str(example['example_id']), question=example['question'],
                   context=ctx, question_type=example.get('question_type'))


def as_draft(draft: Any) -> AnswerDraft:
    """Return a contract ``AnswerDraft``; also accepts the equivalent plain dict
    (E's runner re-reads cached drafts from JSONL as dicts)."""
    if isinstance(draft, AnswerDraft):
        return draft
    if not isinstance(draft, dict):
        raise TypeError(f'expected AnswerDraft or dict, got {type(draft).__name__}')
    return AnswerDraft(draft_id=str(draft['draft_id']), short_answer=draft['short_answer'],
                       explanation=draft.get('explanation') or '')


# --------------------------------------------------------------------------- draft

def generate_draft(example: QAInput, llm: LLMClient, config: AgentConfig) -> AnswerDraft:
    """Create the shared initial draft for one example (one LLM call + optional repair)."""
    example = as_qa_input(example)
    prompt = load_prompt('answer.draft')
    messages = prompt.render(question=example.question.strip(),
                             context=render_context(example.context))
    short, expl = call_and_parse(llm, stage=STAGE_DRAFT, messages=messages,
                                 config=config, parser=parse_answer_payload)
    return AnswerDraft(draft_id=draft_id_for(example), short_answer=short, explanation=expl)


# --------------------------------------------------------------------------- port adapter

class AnswerAgent:
    """Implements ``contracts.interfaces.AnswerPort`` for E's runner.

    >>> agent = AnswerAgent(AgentConfig(model='openai/gpt-4o-mini'))
    >>> draft = agent.generate_draft(example, llm)                 # shared by M1-M5
    >>> m2 = agent.self_reflect(example, draft, llm)               # M2 RevisionResult
    >>> fb = agent.independent_critique(example, draft, llm)       # M3 feedback only
    >>> final = revision_port.revise_answer(example, draft, fb, llm)   # D, called by E
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    @property
    def prompt_versions(self) -> dict[str, str]:
        return dict(PROMPT_VERSIONS)

    @property
    def prompt_version(self) -> dict[str, str]:
        """Alias read by E's runner (``getattr(draft_port, 'prompt_version')``)."""
        return dict(PROMPT_VERSIONS)

    def generate_draft(self, example: QAInput, llm: LLMClient) -> AnswerDraft:
        return generate_draft(as_qa_input(example), llm, self.config)

    def self_reflect(self, example: QAInput, draft: AnswerDraft,
                     llm: LLMClient) -> RevisionResult:
        from .baselines import self_reflect
        return self_reflect(as_qa_input(example), as_draft(draft), llm, self.config)

    def self_reflect_with_feedback(self, example: QAInput, draft: AnswerDraft,
                                   llm: LLMClient) -> tuple[VerificationFeedback, RevisionResult]:
        """M2 with its self-check kept: returns (self-check feedback, revision).

        Same two LLM calls as ``self_reflect``. Offered so E can store M2 feedback in
        ExperimentRecord.feedback (HANDOFF_A issue 2); not part of the v0.1 port.
        """
        from .baselines import run_m2
        out = run_m2(as_qa_input(example), as_draft(draft), llm, self.config)
        return out.feedback, out.final

    def independent_critique(self, example: QAInput, draft: AnswerDraft,
                             llm: LLMClient) -> VerificationFeedback:
        from .independent_critic import independent_critique
        return independent_critique(as_qa_input(example), as_draft(draft), llm, self.config)
