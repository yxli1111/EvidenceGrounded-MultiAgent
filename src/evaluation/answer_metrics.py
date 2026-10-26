"""Official HotpotQA answer evaluation metrics: Exact Match and token F1.

Normalization follows the original HotpotQA / SQuAD evaluation script:
lowercase -> strip punctuation -> remove articles -> collapse whitespace.
"""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def exact_match(prediction: str, gold: str) -> float:
    """Return 1.0 if normalized strings are identical, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 over normalized whitespace-split tokens."""
    pred_tokens = _get_tokens(prediction)
    gold_tokens = _get_tokens(gold)

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
        pred = rec.get("final_answer", "")
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
