"""Official HotpotQA answer evaluation metrics: Exact Match and token F1.

Normalization follows the original HotpotQA / SQuAD evaluation script:
lowercase -> strip punctuation -> remove articles -> collapse whitespace.

Special case (hotpot_evaluate_v1.py): when either the prediction or the gold
answer is one of {'yes', 'no', 'noanswer'}, token-level F1 is not computed;
the score is 1.0 iff the strings match exactly after normalization, else 0.0.
None inputs are treated as empty strings.
"""

import csv
import json
import pathlib
import re
import string
from collections import Counter

_YES_NO_SPECIAL = {"yes", "no", "noanswer"}


def _gold_get(gold, key, default=None):
    """Get a field from a gold entry that may be a dict or a QAGold object."""
    if isinstance(gold, dict):
        return gold.get(key, default)
    return getattr(gold, key, default)


def normalize_answer(s: str | None) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _get_tokens(s: str | None) -> list[str]:
    return normalize_answer(s).split()


def exact_match(prediction: str | None, gold: str | None) -> float:
    """Return 1.0 if normalized strings are identical, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str | None, gold: str | None) -> float:
    """Token-level F1 over normalized whitespace-split tokens.

    Follows hotpot_evaluate_v1.py: if either normalized answer is one of
    {yes, no, noanswer}, skip token comparison and return exact match instead.
    """
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    # HotpotQA yes/no/noanswer special case
    if pred_norm in _YES_NO_SPECIAL or gold_norm in _YES_NO_SPECIAL:
        return 1.0 if pred_norm == gold_norm else 0.0

    pred_tokens = pred_norm.split() if pred_norm else []
    gold_tokens = gold_norm.split() if gold_norm else []

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _get_pred(rec) -> str:
    """Extract the best available prediction from a record (dict or object).

    Prefers final["final_answer"] (new nested structure), falls back to
    draft["short_answer"] (also new nested), then legacy flat fields.
    M1 has final=None by design; falls back to draft.short_answer.
    """
    if isinstance(rec, dict):
        final = rec.get("final") or {}
        draft = rec.get("draft") or {}
        if isinstance(final, dict) and final.get("final_answer"):
            return final["final_answer"]
        if isinstance(draft, dict) and draft.get("short_answer"):
            return draft["short_answer"]
        # Legacy flat field fallback
        return rec.get("final_answer") or rec.get("short_answer") or ""
    else:  # ExperimentRecord object
        final = getattr(rec, "final", None) or {}
        draft = getattr(rec, "draft", None) or {}
        if isinstance(final, dict) and final.get("final_answer"):
            return final["final_answer"]
        if isinstance(draft, dict) and draft.get("short_answer"):
            return draft["short_answer"]
        # Legacy flat field fallback
        return getattr(rec, "final_answer", None) or getattr(rec, "short_answer", None) or ""


def _get_draft_pred(rec) -> str:
    """Extract the draft short_answer for draft_em computation."""
    if isinstance(rec, dict):
        draft = rec.get("draft") or {}
        if isinstance(draft, dict) and draft.get("short_answer"):
            return draft["short_answer"]
        # Legacy flat field fallback
        return rec.get("short_answer") or ""
    else:
        draft = getattr(rec, "draft", None) or {}
        if isinstance(draft, dict) and draft.get("short_answer"):
            return draft["short_answer"]
        # Legacy flat field fallback
        return getattr(rec, "short_answer", None) or ""


def _feedback_counts(feedback) -> tuple[int, int, int]:
    if not feedback:
        return 0, 0, 0
    items = (feedback.get("items") if isinstance(feedback, dict) else getattr(feedback, "items", [])) or []
    def _label(i):
        return i.get("label") if isinstance(i, dict) else getattr(i, "label", None)
    return (
        sum(1 for i in items if _label(i) == "SUPPORTED"),
        sum(1 for i in items if _label(i) == "UNSUPPORTED"),
        sum(1 for i in items if _label(i) == "CONTRADICTED"),
    )


def _answer_changed(rec) -> str:
    mid = rec.get("method_id", "") if isinstance(rec, dict) else getattr(rec, "method_id", "")
    if mid == "M1":
        return ""
    status = rec.get("status", "") if isinstance(rec, dict) else getattr(rec, "status", "")
    if status != "ok":
        return ""
    final = (rec.get("final") if isinstance(rec, dict) else getattr(rec, "final", None)) or {}
    draft = (rec.get("draft") if isinstance(rec, dict) else getattr(rec, "draft", None)) or {}
    if not final or not draft:
        return ""
    fa = final.get("final_answer", "") if isinstance(final, dict) else getattr(final, "final_answer", "")
    sa = draft.get("short_answer", "") if isinstance(draft, dict) else getattr(draft, "short_answer", "")
    return "1" if fa != sa else "0"


def _explanation_changed(rec) -> str:
    mid = rec.get("method_id", "") if isinstance(rec, dict) else getattr(rec, "method_id", "")
    if mid == "M1":
        return ""
    status = rec.get("status", "") if isinstance(rec, dict) else getattr(rec, "status", "")
    if status != "ok":
        return ""
    final = (rec.get("final") if isinstance(rec, dict) else getattr(rec, "final", None)) or {}
    draft = (rec.get("draft") if isinstance(rec, dict) else getattr(rec, "draft", None)) or {}
    if not final or not draft:
        return ""
    fe = final.get("final_explanation", "") if isinstance(final, dict) else getattr(final, "final_explanation", "")
    de = draft.get("explanation", "") if isinstance(draft, dict) else getattr(draft, "explanation", "")
    return "1" if fe != de else "0"


def _sf_recall(rec, gold_entry) -> str:
    gold_sfs = _gold_get(gold_entry, "gold_supporting_facts") if gold_entry else None
    if not gold_sfs:
        return ""
    evidence = (rec.get("evidence") if isinstance(rec, dict) else getattr(rec, "evidence", None)) or {}
    if not isinstance(evidence, dict):
        return ""
    retrieved: set[tuple] = set()
    for ev_list in evidence.values():
        for ev in (ev_list or []):
            if isinstance(ev, dict):
                retrieved.add((ev.get("doc_title", ""), ev.get("sentence_id", -1)))
            else:
                retrieved.add((getattr(ev, "doc_title", ""), getattr(ev, "sentence_id", -1)))
    if not retrieved:
        return ""
    n_hit = sum(
        1 for sf in gold_sfs
        if isinstance(sf, dict) and (sf.get("title", ""), sf.get("sentence_id", -1)) in retrieved
    )
    return f"{n_hit / len(gold_sfs):.4f}"


def compute_em_f1_batch(
    records: list[dict],
    gold_lookup: dict[str, dict],
) -> dict[str, dict]:
    """Compute per-condition EM and F1 from a list of ExperimentRecord dicts.

    Args:
        records: list of ExperimentRecord dicts with keys
                 example_id, method_id, status, and either nested
                 final/draft dicts or legacy flat fields.
                 Records with status != "ok" count as 0 in both numerator
                 and denominator (failure stays in denominator).
        gold_lookup: maps example_id -> {"gold_answer": str, ...}

    Returns:
        dict keyed by method_id with {"em": float, "f1": float, "n": int,
        "n_ok": int, "n_error": int}.
    """
    from collections import defaultdict

    # Dedup: keep highest attempt per (example_id, method_id)
    seen: dict[tuple, dict] = {}
    for rec in records:
        key = (
            rec.get("example_id") if isinstance(rec, dict) else getattr(rec, "example_id", ""),
            rec.get("method_id") if isinstance(rec, dict) else getattr(rec, "method_id", ""),
        )
        attempt = rec.get("attempt", 1) if isinstance(rec, dict) else getattr(rec, "attempt", 1)
        if key not in seen or attempt > (
            seen[key].get("attempt", 1) if isinstance(seen[key], dict) else getattr(seen[key], "attempt", 1)
        ):
            seen[key] = rec
    deduped = list(seen.values())

    buckets: dict[str, dict] = defaultdict(
        lambda: {"em_sum": 0.0, "f1_sum": 0.0, "n": 0, "n_ok": 0, "n_error": 0}
    )

    for rec in deduped:
        mid = rec["method_id"] if isinstance(rec, dict) else getattr(rec, "method_id", "")
        eid = rec["example_id"] if isinstance(rec, dict) else getattr(rec, "example_id", "")
        status = rec.get("status") if isinstance(rec, dict) else getattr(rec, "status", "error")
        bucket = buckets[mid]
        bucket["n"] += 1

        if status != "ok":
            bucket["n_error"] += 1
            # failed sample contributes 0 to numerator, still counted in denominator
            continue

        bucket["n_ok"] += 1
        gold_entry = gold_lookup.get(eid)
        gold = _gold_get(gold_entry, "gold_answer", "") if gold_entry is not None else ""
        pred = _get_pred(rec)
        bucket["em_sum"] += exact_match(pred, gold)
        bucket["f1_sum"] += token_f1(pred, gold)

    result = {}
    for mid, b in buckets.items():
        n = b["n"]
        result[mid] = {
            "em": b["em_sum"] / n if n > 0 else 0.0,
            "f1": b["f1_sum"] / n if n > 0 else 0.0,
            "n": n,
            "n_ok": b["n_ok"],
            "n_error": b["n_error"],
        }
    return result


def export_per_example_csv(
    records: list[dict],
    gold_lookup: dict[str, dict],
    output_path: str = "results/evaluation/per_example.csv",
    preliminary: bool = True,
    omitted_metrics: list[str] | None = None,
    split: str | None = None,
    manifest: str | None = None,
    extra_meta: dict | None = None,
) -> str:
    """Export per-example EM/F1 scores to a CSV file.

    Deduplicates records by keeping the highest-attempt record per
    (example_id, method_id) pair so that retry runs are not double-counted.

    Writes a metadata.json file alongside the CSV containing the preliminary
    flag, sample count (n), and optional split/manifest/omitted_metrics info.

    Args:
        records: list of ExperimentRecord dicts.
        gold_lookup: maps example_id -> {"gold_answer": str, ...}.
        output_path: destination path (created with parent dirs as needed).
        preliminary: flag written to metadata.json.
        omitted_metrics: list of metric names omitted from this run.
        split: optional split label for metadata.json.
        manifest: optional manifest file path for metadata.json.

    Returns:
        Absolute path of the written CSV file.

    Columns: example_id, method_id, status, em, f1, draft_em, question_type,
             method_cost_usd, draft_cost_usd, n_calls, latency_s,
             n_evidence, n_claims, n_supported, n_unsupported, n_contradicted,
             answer_changed, explanation_changed, sf_recall, error
    """
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Dedup: keep highest attempt per (example_id, method_id)
    seen: dict = {}
    for rec in records:
        eid = rec.get("example_id", "") if isinstance(rec, dict) else getattr(rec, "example_id", "")
        mid = rec.get("method_id", "") if isinstance(rec, dict) else getattr(rec, "method_id", "")
        attempt = (rec.get("attempt", 1) if isinstance(rec, dict) else getattr(rec, "attempt", 1)) or 1
        key = (eid, mid)
        existing = seen.get(key)
        if existing is None:
            seen[key] = rec
        else:
            ex_attempt = (existing.get("attempt", 1) if isinstance(existing, dict) else getattr(existing, "attempt", 1)) or 1
            if attempt > ex_attempt:
                seen[key] = rec
    deduped = list(seen.values())

    # Write metadata.json alongside the CSV
    n_examples = len({
        (rec.get("example_id", "") if isinstance(rec, dict) else getattr(rec, "example_id", ""))
        for rec in deduped
    })
    meta: dict = {
        "preliminary": preliminary,
        "n_examples": n_examples,
        "n_records": len(deduped),
    }
    if omitted_metrics is not None:
        meta["omitted_metrics"] = omitted_metrics
    if split is not None:
        meta["split"] = split
    if manifest is not None:
        meta["manifest"] = manifest
    if extra_meta:
        meta.update(extra_meta)
    meta_path = out.parent / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "example_id", "method_id", "status", "em", "f1", "draft_em", "question_type",
            "method_cost_usd", "draft_cost_usd", "n_calls", "latency_s",
            "n_evidence", "n_claims", "n_supported", "n_unsupported", "n_contradicted",
            "answer_changed", "explanation_changed", "sf_recall", "error",
        ])
        for rec in deduped:
            eid = rec.get("example_id", "") if isinstance(rec, dict) else getattr(rec, "example_id", "")
            mid = rec.get("method_id", "") if isinstance(rec, dict) else getattr(rec, "method_id", "")
            status = rec.get("status", "") if isinstance(rec, dict) else getattr(rec, "status", "")
            qt = rec.get("question_type") if isinstance(rec, dict) else getattr(rec, "question_type", None)

            gold_entry = gold_lookup.get(eid)
            gold_answer = _gold_get(gold_entry, "gold_answer", "") if gold_entry is not None else ""
            if status != "ok":
                em_val = f1_val = draft_em_val = 0.0
            else:
                pred = _get_pred(rec)
                draft_pred = _get_draft_pred(rec)
                em_val = exact_match(pred, gold_answer)
                f1_val = token_f1(pred, gold_answer)
                draft_em_val = exact_match(draft_pred, gold_answer)

            usage = (rec.get("usage") if isinstance(rec, dict) else getattr(rec, "usage", None)) or []
            draft_usage = (rec.get("draft_usage") if isinstance(rec, dict) else getattr(rec, "draft_usage", None)) or []
            evidence = (rec.get("evidence") if isinstance(rec, dict) else getattr(rec, "evidence", None)) or {}
            claims = (rec.get("claims") if isinstance(rec, dict) else getattr(rec, "claims", None))
            feedback = (rec.get("feedback") if isinstance(rec, dict) else getattr(rec, "feedback", None))
            error_msg = (rec.get("error") if isinstance(rec, dict) else getattr(rec, "error", None)) or ""

            method_cost = sum((u.get("cost_usd") or 0.0) for u in usage if isinstance(u, dict))
            draft_cost = sum((u.get("cost_usd") or 0.0) for u in draft_usage if isinstance(u, dict))
            n_calls = len(usage)
            latency = sum((u.get("latency_seconds") or 0.0) for u in usage if isinstance(u, dict))
            n_evidence = sum(len(v) for v in evidence.values() if isinstance(v, list)) if isinstance(evidence, dict) else 0
            n_claims_val = len(claims) if claims else 0
            n_sup, n_uns, n_con = _feedback_counts(feedback)

            writer.writerow([
                eid, mid, status,
                f"{em_val:.4f}", f"{f1_val:.4f}", f"{draft_em_val:.4f}", qt or "",
                f"{method_cost:.6f}", f"{draft_cost:.6f}", n_calls, f"{latency:.3f}",
                n_evidence, n_claims_val, n_sup, n_uns, n_con,
                _answer_changed(rec), _explanation_changed(rec), _sf_recall(rec, gold_entry),
                error_msg,
            ])

    return str(out.resolve())
