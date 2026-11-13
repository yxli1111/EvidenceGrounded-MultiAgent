"""Small validation and prompt helpers shared by D's three public functions."""

import hashlib
import json
import math
from pathlib import Path

from .config import PROMPT_VERSIONS, settings
from .schema_adapter import make_feedback


LABELS = {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED"}
_ROOT = Path(__file__).resolve().parents[2]


def text(value, name: str, error, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise error(f"{name} must be {'a string' if allow_empty else 'a non-empty string'}.")
    return value


def sequence(value, name: str, error) -> list:
    if not isinstance(value, (list, tuple)):
        raise error(f"{name} must be a list or tuple.")
    return list(value)


def nonnegative_int(value, name: str, error) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise error(f"{name} must be a non-negative integer.")
    return value


def example_payload(example, error) -> tuple[dict, dict]:
    """Explicit allowlist: gold or extra attributes never enter an LLM prompt."""
    context = []
    by_key = {}
    for sentence in sequence(example.context, "context", error):
        title = text(sentence.title, "sentence.title", error)
        index = nonnegative_int(sentence.sentence_id, "sentence.sentence_id", error)
        body = text(sentence.text, "sentence.text", error, allow_empty=True)
        key = (title, index)
        if key in by_key:
            raise error("Context contains a duplicate (title, sentence_id).")
        by_key[key] = body
        context.append({"title": title, "sentence_id": index, "text": body})
    return {
        "example_id": text(example.example_id, "example_id", error),
        "question": text(example.question, "question", error),
        "context": context,
    }, by_key


def draft_payload(draft, error) -> dict:
    return {
        "draft_id": text(draft.draft_id, "draft_id", error),
        "short_answer": text(draft.short_answer, "short_answer", error, allow_empty=True),
        "explanation": text(draft.explanation, "explanation", error, allow_empty=True),
    }


def evidence_payload(evidence, context_by_key: dict, error) -> list[dict]:
    result = []
    seen = set()
    for item in sequence(evidence, "evidence", error):
        title = text(item.doc_title, "evidence.doc_title", error)
        index = nonnegative_int(item.sentence_id, "evidence.sentence_id", error)
        key = (title, index)
        body = text(item.text, "evidence.text", error, allow_empty=True)
        if key not in context_by_key or context_by_key[key] != body:
            raise error("Evidence title/index/text does not match the supplied context.")
        if key in seen:
            raise error("A target's evidence contains duplicate sentences.")
        seen.add(key)
        score = item.score
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise error("evidence.score must be a finite number.")
        rank = nonnegative_int(item.rank, "evidence.rank", error)
        if rank == 0:
            raise error("evidence.rank is one-based.")
        result.append({
            "evidence_id": f"{title}::{index}",
            "doc_title": title,
            "sentence_id": index,
            "text": body,
            "score": float(score),
            "retrieval_query": text(item.retrieval_query, "retrieval_query", error, allow_empty=True),
            "rank": rank,
        })
    return result


def generate_json(llm, stage: str, payload: dict, error) -> dict:
    config = settings()
    folder = "revision" if stage == "revise_answer" else "verification"
    prompt = (_ROOT / "prompts" / folder / f"{stage}.txt").read_text(encoding="utf-8")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)},
    ]
    call_args = {
        "stage": stage,
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": (config.revision_max_tokens if stage == "revise_answer"
                       else config.verification_max_tokens),
    }
    # Transport retries and usage/raw-response logging belong to the LLM client.
    # Only verify_draft may request one syntax-only JSON repair; never retry
    # API failures, semantic/schema errors, citations, or revisions here.
    response = llm.generate(messages=messages, **call_args)
    raw = text(response.text, "LLMResponse.text", error)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        if stage != "verify_draft":
            raise error(f"{stage}: response is not valid JSON ({exc.msg}).") from exc
        repair_messages = [
            {"role": "system", "content": (
                "Repair only JSON syntax in the supplied verifier output. "
                "Return exactly one valid JSON object, with no Markdown. "
                "Preserve all verdicts, target IDs, rationales, citation IDs, "
                "suggested corrections, and overall comment exactly. "
                "Do not re-verify, invent facts, or add citations."
            )},
            {"role": "user", "content": raw},
        ]
        repaired = llm.generate(messages=repair_messages, **call_args)
        repair_raw = text(repaired.text, "LLMResponse.text", error)
        try:
            parsed = json.loads(repair_raw)
        except json.JSONDecodeError as repair_exc:
            raise error(
                f"{stage}: JSON syntax repair failed ({repair_exc.msg})."
            ) from repair_exc
    if not isinstance(parsed, dict):
        raise error(f"{stage}: response must be a JSON object.")
    return parsed


def feedback_result(draft_id: str, stage: str, items: list[dict], comment: str):
    identity = json.dumps(
        {"draft_id": draft_id, "stage": stage, "version": PROMPT_VERSIONS[stage],
         "items": items, "overall_comment": comment},
        sort_keys=True, ensure_ascii=False, allow_nan=False,
    )
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return make_feedback(f"{draft_id}:{stage}:{suffix}", items, comment)
