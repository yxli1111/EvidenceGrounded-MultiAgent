"""Cost and usage aggregation from ExperimentRecords.

Terminology
-----------
draft_usage   – the generate_draft LLM call, shared across all five methods
                for the same example.  Kept separate to avoid 5× double-counting.
usage         – method-specific LLM calls (self_reflect, critique, verify_draft,
                verify_claims, extract_claims, revise_answer).

Rules
-----
* Never fabricate numbers: if cost_usd is None in a call record, it is excluded
  from sums and the missing-data count is incremented.
* Error records (status != "ok") contribute 0 to usage sums but are included in
  n_records so denominators are never silently dropped.
* Unit prices are loaded from a JSON config file; they are never hardcoded here.
  Pass pricing=None to skip token-based re-estimation and use recorded cost only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DEFAULT_PRICING_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "model_pricing.json"
)


# ---------------------------------------------------------------------------
# Pricing config loader
# ---------------------------------------------------------------------------

def load_pricing(path: str | Path | None = None) -> dict[str, dict]:
    """Load model unit prices from a JSON config file.

    Returns:
        dict keyed by model_id ->
          {"input_per_1k_tokens": float, "output_per_1k_tokens": float}
        Returns {} if file is missing or malformed (no crash; caller decides).
    """
    target = Path(path) if path is not None else _DEFAULT_PRICING_PATH
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return {k: dict(v) for k, v in data.get("models", {}).items()}
    except (FileNotFoundError, json.JSONDecodeError, AttributeError):
        return {}


def compute_cost_from_tokens(
    input_tokens: int | None,
    output_tokens: int | None,
    model: str,
    pricing: dict[str, dict],
) -> float | None:
    """Re-derive cost from token counts and unit prices.

    Returns None if tokens or pricing entry are unavailable.
    """
    entry = pricing.get(model)
    if not entry:
        return None
    if input_tokens is None or output_tokens is None:
        return None
    return (
        input_tokens / 1000.0 * entry["input_per_1k_tokens"]
        + output_tokens / 1000.0 * entry["output_per_1k_tokens"]
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _usage_lists(record: Any, include_draft: bool) -> list[list[dict]]:
    lists = [_get(record, "usage") or []]
    if include_draft:
        lists.append(_get(record, "draft_usage") or [])
    return lists


def _accumulate(
    acc: dict,
    calls: list[dict],
    pricing: dict[str, dict],
) -> None:
    """Add one list of call dicts into accumulator in-place."""
    for call in calls:
        acc["llm_calls"] += 1

        in_tok = call.get("input_tokens")
        out_tok = call.get("output_tokens")
        cost = call.get("cost_usd")
        lat = call.get("latency_ms")
        model = call.get("model", "")
        cache = call.get("cache_hit")

        if in_tok is not None:
            acc["input_tokens_sum"] = (acc["input_tokens_sum"] or 0) + in_tok
        if out_tok is not None:
            acc["output_tokens_sum"] = (acc["output_tokens_sum"] or 0) + out_tok

        # Use recorded cost if available; otherwise re-compute from tokens.
        if cost is None and pricing:
            cost = compute_cost_from_tokens(in_tok, out_tok, model, pricing)
        if cost is not None:
            acc["cost_usd_sum"] = (acc["cost_usd_sum"] or 0.0) + cost
            acc["n_calls_with_cost"] += 1

        if lat is not None:
            acc["_latency_sum"] += lat
            acc["n_calls_with_latency"] += 1

        if cache is True:
            acc["cache_hits"] += 1


def _blank_acc() -> dict:
    return {
        "n_records": 0,
        "n_ok": 0,
        "n_error": 0,
        "llm_calls": 0,
        "input_tokens_sum": None,
        "output_tokens_sum": None,
        "cost_usd_sum": None,
        "n_calls_with_cost": 0,
        "_latency_sum": 0.0,
        "n_calls_with_latency": 0,
        "cache_hits": 0,
    }


def _finalise(acc: dict, method_id: str) -> dict:
    n_ok = acc["n_ok"]
    mean_lat = (
        acc["_latency_sum"] / acc["n_calls_with_latency"]
        if acc["n_calls_with_latency"] > 0 else None
    )
    per_ex = (
        acc["cost_usd_sum"] / n_ok
        if n_ok > 0 and acc["cost_usd_sum"] is not None else None
    )
    return {
        "method_id": method_id,
        "n_records": acc["n_records"],
        "n_ok": n_ok,
        "n_error": acc["n_error"],
        "llm_calls": acc["llm_calls"],
        "input_tokens_sum": acc["input_tokens_sum"],
        "output_tokens_sum": acc["output_tokens_sum"],
        "cost_usd_sum": acc["cost_usd_sum"],
        "n_calls_with_cost": acc["n_calls_with_cost"],
        "mean_latency_ms": mean_lat,
        "n_calls_with_latency": acc["n_calls_with_latency"],
        "cache_hits": acc["cache_hits"],
        "per_example_cost_usd": per_ex,
    }


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def aggregate_usage_by_method(
    records: list,
    include_draft_usage: bool = False,
    pricing: dict[str, dict] | None = None,
) -> dict[str, dict]:
    """Aggregate LLM usage stats from ExperimentRecords, keyed by method_id.

    Args:
        records: list of ExperimentRecord objects or dicts.
        include_draft_usage: if True, add draft_usage to each method's totals.
            Keep False for method-to-method cost comparison (avoids 5× inflate).
        pricing: model unit prices from load_pricing(); used to re-derive cost
            when cost_usd is None in a call record.  Pass None to skip.

    Returns:
        dict[method_id, summary_dict].  Methods with zero records are omitted.
    """
    if pricing is None:
        pricing = {}
    buckets: dict[str, dict] = {}

    for rec in records:
        mid = _get(rec, "method_id") or "UNKNOWN"
        if mid not in buckets:
            buckets[mid] = _blank_acc()
        acc = buckets[mid]
        acc["n_records"] += 1

        status = _get(rec, "status") or "error"
        if status != "ok":
            acc["n_error"] += 1
            continue

        acc["n_ok"] += 1
        for calls in _usage_lists(rec, include_draft_usage):
            _accumulate(acc, calls, pricing)

    return {mid: _finalise(acc, mid) for mid, acc in buckets.items()}


def aggregate_draft_usage(
    records: list,
    pricing: dict[str, dict] | None = None,
) -> dict:
    """Aggregate the shared generate_draft usage across all records.

    Counts only one draft_usage entry per example (uses first record per
    example_id found in the list).  This avoids 5× counting when all five
    method records for the same example share the same draft.

    Returns:
        Summary dict with keys: n_examples, llm_calls, input_tokens_sum,
        output_tokens_sum, cost_usd_sum, n_calls_with_cost, mean_latency_ms,
        cache_hits.
    """
    if pricing is None:
        pricing = {}
    seen: set[str] = set()
    acc = _blank_acc()

    for rec in records:
        eid = _get(rec, "example_id") or ""
        if eid in seen:
            continue
        seen.add(eid)
        draft_calls = _get(rec, "draft_usage") or []
        _accumulate(acc, draft_calls, pricing)

    n_ex = len(seen)
    mean_lat = (
        acc["_latency_sum"] / acc["n_calls_with_latency"]
        if acc["n_calls_with_latency"] > 0 else None
    )
    return {
        "n_examples": n_ex,
        "llm_calls": acc["llm_calls"],
        "input_tokens_sum": acc["input_tokens_sum"],
        "output_tokens_sum": acc["output_tokens_sum"],
        "cost_usd_sum": acc["cost_usd_sum"],
        "n_calls_with_cost": acc["n_calls_with_cost"],
        "mean_latency_ms": mean_lat,
        "cache_hits": acc["cache_hits"],
    }


def estimate_budget(
    trial_records: list,
    target_n_examples: int,
    pricing: dict[str, dict] | None = None,
) -> dict:
    """Estimate full-run cost by scaling trial per-example averages.

    The estimator uses the average per-example method cost from the trial
    (ok records only).  Failed examples reduce the ok-count and thus inflate
    the per-example estimate slightly — conservative, not optimistic.

    Args:
        trial_records: ExperimentRecords from a completed trial run.
        target_n_examples: planned number of examples in the full study.
        pricing: from load_pricing(); used to fill in None costs.

    Returns:
        {
          "trial_n_examples": int,
          "target_n_examples": int,
          "draft": {"trial_cost_usd": ..., "estimated_cost_usd": ...},
          "methods": {method_id: {
              "trial_cost_usd": float | None,
              "per_example_cost_usd": float | None,
              "estimated_cost_usd": float | None,
              "trial_calls": int,
              "estimated_calls": int,
          }},
          "total_estimated_cost_usd": float | None,
        }
    """
    if pricing is None:
        pricing = {}

    by_method = aggregate_usage_by_method(trial_records, pricing=pricing)
    draft_agg = aggregate_draft_usage(trial_records, pricing=pricing)
    trial_n = draft_agg["n_examples"]

    scale = target_n_examples / trial_n if trial_n > 0 else 0.0

    methods_out: dict[str, dict] = {}
    total_est: float | None = 0.0

    for mid, agg in by_method.items():
        trial_cost = agg["cost_usd_sum"]
        per_ex = agg["per_example_cost_usd"]
        est_cost = (per_ex * target_n_examples) if per_ex is not None else None
        n_ok = agg["n_ok"]
        calls_per_ex = agg["llm_calls"] / n_ok if n_ok > 0 else 0.0
        methods_out[mid] = {
            "trial_cost_usd": trial_cost,
            "per_example_cost_usd": per_ex,
            "estimated_cost_usd": est_cost,
            "trial_calls": agg["llm_calls"],
            "estimated_calls": round(calls_per_ex * target_n_examples),
        }
        if est_cost is None:
            total_est = None
        elif total_est is not None:
            total_est += est_cost

    draft_cost = draft_agg["cost_usd_sum"]
    draft_per_ex = (draft_cost / trial_n) if (draft_cost is not None and trial_n > 0) else None
    draft_est = (draft_per_ex * target_n_examples) if draft_per_ex is not None else None
    if draft_est is None:
        total_est = None
    elif total_est is not None:
        total_est += draft_est

    return {
        "trial_n_examples": trial_n,
        "target_n_examples": target_n_examples,
        "draft": {
            "trial_cost_usd": draft_cost,
            "per_example_cost_usd": draft_per_ex,
            "estimated_cost_usd": draft_est,
        },
        "methods": methods_out,
        "total_estimated_cost_usd": total_est,
    }
