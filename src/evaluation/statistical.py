"""Paired bootstrap confidence intervals for comparing two methods.

Algorithm
---------
For two sequences of per-example scores A[i] and B[i] (same examples):
  1. Compute the observed difference: delta_obs = mean(A) - mean(B).
  2. Repeat R times:
       - Sample n indices with replacement (bootstrap sample).
       - delta_b = mean(A[indices]) - mean(B[indices]).
  3. CI = [percentile(alpha/2), percentile(1 - alpha/2)] of delta_b.
  4. p-value (one-sided, H0: delta <= 0): fraction of bootstrap deltas <= 0.

References: Efron & Tibshirani (1993); Berg et al. ACL 2012 bootstrap test.
"""

from __future__ import annotations

import random
from typing import Sequence


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = 42,
) -> dict:
    """Paired bootstrap CI for mean(A) - mean(B).

    Args:
        scores_a: Per-example scores for method A (length n).
        scores_b: Per-example scores for method B (same length, same order).
        n_resamples: Bootstrap iterations (default 10 000).
        confidence: Desired confidence level (default 0.95 → 95 % CI).
        seed: RNG seed for reproducibility; None = no seeding.

    Returns:
        {
          "n": int,                    # number of paired examples
          "delta_obs": float,          # mean(A) - mean(B) on observed data
          "ci_low": float,             # lower bound of CI
          "ci_high": float,            # upper bound of CI
          "confidence": float,         # requested confidence level
          "p_value": float,            # P(delta <= 0) under bootstrap null
          "n_resamples": int,
        }

    Raises:
        ValueError: if sequences differ in length or are empty.
    """
    a = list(scores_a)
    b = list(scores_b)
    n = len(a)
    if n == 0:
        raise ValueError("scores_a and scores_b must be non-empty")
    if len(b) != n:
        raise ValueError(
            f"scores_a and scores_b must have the same length ({n} vs {len(b)})"
        )

    rng = random.Random(seed)
    delta_obs = sum(a) / n - sum(b) / n

    bootstrap_deltas: list[float] = []
    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        da = sum(a[i] for i in indices) / n
        db = sum(b[i] for i in indices) / n
        bootstrap_deltas.append(da - db)

    bootstrap_deltas.sort()
    alpha = 1.0 - confidence
    lo_idx = max(0, int(alpha / 2 * n_resamples))
    hi_idx = min(n_resamples - 1, int((1 - alpha / 2) * n_resamples))

    p_value = sum(1 for d in bootstrap_deltas if d <= 0) / n_resamples

    return {
        "n": n,
        "delta_obs": delta_obs,
        "ci_low": bootstrap_deltas[lo_idx],
        "ci_high": bootstrap_deltas[hi_idx],
        "confidence": confidence,
        "p_value": p_value,
        "n_resamples": n_resamples,
    }


def compare_methods(
    records: list,
    gold_lookup: dict,
    method_a: str,
    method_b: str,
    metric: str = "em",
    **bootstrap_kwargs,
) -> dict:
    """Run paired_bootstrap_ci on records from two named methods.

    Failed records (status != "ok") are included with score 0.0.
    Missing records for one method on a given example also score 0.0.
    An example is excluded only if NEITHER method has any record for it.

    Args:
        records: list of ExperimentRecord dicts (or objects with .example_id etc).
        gold_lookup: maps example_id → {"gold_answer": str, ...}.
        method_a: method_id for group A.
        method_b: method_id for group B.
        metric: "em" or "f1".
        **bootstrap_kwargs: forwarded to paired_bootstrap_ci.

    Returns:
        paired_bootstrap_ci result dict with extra keys "method_a", "method_b",
        "metric", "mean_a", "mean_b", "n_failed_a", "n_failed_b".
    """
    from .answer_metrics import exact_match, token_f1, _gold_get

    def _get(obj, key):
        return obj[key] if isinstance(obj, dict) else getattr(obj, key, None)

    def _score_ok(rec) -> float:
        """Compute score for an ok record; returns 0.0 if gold missing."""
        eid = _get(rec, "example_id") or ""
        gold_entry = gold_lookup.get(eid)
        if not gold_entry:
            return 0.0
        gold = _gold_get(gold_entry, "gold_answer", "")
        # Handle nested field structure (Phase 2): final/draft dicts
        if isinstance(rec, dict):
            final = rec.get("final") or {}
            draft = rec.get("draft") or {}
            if isinstance(final, dict) and final.get("final_answer"):
                pred = final["final_answer"]
            elif isinstance(draft, dict) and draft.get("short_answer"):
                pred = draft["short_answer"]
            else:
                pred = rec.get("final_answer") or ""
        else:
            final = getattr(rec, "final", None) or {}
            draft = getattr(rec, "draft", None) or {}
            if isinstance(final, dict) and final.get("final_answer"):
                pred = final["final_answer"]
            elif isinstance(draft, dict) and draft.get("short_answer"):
                pred = draft["short_answer"]
            else:
                pred = getattr(rec, "final_answer", None) or ""
        if metric == "f1":
            return token_f1(pred, gold)
        return exact_match(pred, gold)

    # Gather per-example scores (or None if no record exists for that method)
    a_records: dict[str, tuple[float, bool]] = {}  # eid → (score, is_failed)
    b_records: dict[str, tuple[float, bool]] = {}
    for rec in records:
        mid = _get(rec, "method_id") or ""
        eid = _get(rec, "example_id") or ""
        status = _get(rec, "status") or "error"
        is_failed = (status != "ok")
        score = 0.0 if is_failed else _score_ok(rec)
        if mid == method_a:
            a_records[eid] = (score, is_failed)
        elif mid == method_b:
            b_records[eid] = (score, is_failed)

    # Include every example that appears in at least one method's records
    all_eids = sorted(a_records.keys() | b_records.keys())
    if not all_eids:
        raise ValueError(
            f"No paired examples found for methods '{method_a}' and '{method_b}'"
        )

    a_vec: list[float] = []
    b_vec: list[float] = []
    n_failed_a = 0
    n_failed_b = 0
    for eid in all_eids:
        a_score, a_fail = a_records.get(eid, (0.0, True))
        b_score, b_fail = b_records.get(eid, (0.0, True))
        a_vec.append(a_score)
        b_vec.append(b_score)
        if a_fail:
            n_failed_a += 1
        if b_fail:
            n_failed_b += 1

    result = paired_bootstrap_ci(a_vec, b_vec, **bootstrap_kwargs)
    result["method_a"] = method_a
    result["method_b"] = method_b
    result["metric"] = metric
    result["mean_a"] = sum(a_vec) / len(a_vec)
    result["mean_b"] = sum(b_vec) / len(b_vec)
    result["n_failed_a"] = n_failed_a
    result["n_failed_b"] = n_failed_b
    return result
