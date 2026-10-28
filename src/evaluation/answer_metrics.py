"""Official HotpotQA answer evaluation metrics: Exact Match and token F1.

Normalization follows the original HotpotQA / SQuAD evaluation script:
lowercase -> strip punctuation -> remove articles -> collapse whitespace.

Special case (hotpot_evaluate_v1.py): when either the prediction or the gold
answer is one of {'yes', 'no', 'noanswer'}, token-level F1 is not computed;
the score is 1.0 iff the strings match exactly after normalization, else 0.0.
None inputs are treated as empty strings.
"""

import csv
import pathlib
import re
import string
from collections import Counter

_YES_NO_SPECIAL = {"yes", "no", "noanswer"}


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


def compute_em_f1_batch(
    records: list[dict],
    gold_lookup: dict[str, dict],
) -> dict[str, dict]:
    """Compute per-condition EM and F1 from a list of ExperimentRecord dicts.

    Args:
        records: list of ExperimentRecord dicts with keys
                 example_id, method_id, status, final_answer.
                 Records with status != "ok" count as 0 in both numerator
                 and denominator (failure stays in denominator).
        gold_lookup: maps example_id -> {"gold_answer": str, ...}

    Returns:
        dict keyed by method_id with {"em": float, "f1": float, "n": int,
        "n_ok": int, "n_error": int}.
    """
    from collections import defaultdict

    buckets: dict[str, dict] = defaultdict(
        lambda: {"em_sum": 0.0, "f1_sum": 0.0, "n": 0, "n_ok": 0, "n_error": 0}
    )

    for rec in records:
        mid = rec["method_id"]
        eid = rec["example_id"]
        bucket = buckets[mid]
        bucket["n"] += 1

        if rec.get("status") != "ok":
            bucket["n_error"] += 1
            # failed sample contributes 0 to numerator, still counted in denominator
            continue

        bucket["n_ok"] += 1
        gold = gold_lookup.get(eid, {}).get("gold_answer", "")
        pred = rec.get("final_answer") or ""
        bucket["em_sum"] += exact_match(pred, gold)
        bucket["f1_sum"] += token_f1(pred, gold)  # pred already "" if None

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
) -> str:
    """Export per-example EM/F1 scores to a CSV file.

    Args:
        records: list of ExperimentRecord dicts.
        gold_lookup: maps example_id -> {"gold_answer": str, ...}.
        output_path: destination path (created with parent dirs as needed).

    Returns:
        Absolute path of the written file.

    Columns: example_id, method_id, status, em, f1, draft_em, question_type
    """
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["example_id", "method_id", "status", "em", "f1",
                          "draft_em", "question_type"])
        for rec in records:
            eid = rec.get("example_id", "") if isinstance(rec, dict) else getattr(rec, "example_id", "")
            mid = rec.get("method_id", "") if isinstance(rec, dict) else getattr(rec, "method_id", "")
            status = rec.get("status", "") if isinstance(rec, dict) else getattr(rec, "status", "")
            qt = rec.get("question_type") if isinstance(rec, dict) else getattr(rec, "question_type", None)

            gold_answer = gold_lookup.get(eid, {}).get("gold_answer", "")
            if status != "ok":
                em_val = f1_val = draft_em_val = 0.0
            else:
                pred = (rec.get("final_answer") if isinstance(rec, dict) else getattr(rec, "final_answer", None)) or ""
                draft_pred = (rec.get("short_answer") if isinstance(rec, dict) else getattr(rec, "short_answer", None)) or ""
                em_val = exact_match(pred, gold_answer)
                f1_val = token_f1(pred, gold_answer)
                draft_em_val = exact_match(draft_pred, gold_answer)

            writer.writerow([eid, mid, status,
                              f"{em_val:.4f}", f"{f1_val:.4f}", f"{draft_em_val:.4f}",
                              qt or ""])

    return str(out.resolve())
