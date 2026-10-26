"""Unit tests for src/evaluation/cost_metrics.py.

All inputs are synthetic; no real data or API calls used.
Prices in the placeholder config are 0.0 for mock-model-v1, so tests that
need non-zero costs either inject a custom pricing dict or use a different
model name.
"""

import json
import os
import tempfile

import pytest

from src.evaluation.cost_metrics import (
    aggregate_draft_usage,
    aggregate_usage_by_method,
    compute_cost_from_tokens,
    estimate_budget,
    load_pricing,
)


# ---------------------------------------------------------------------------
# Synthetic record builders
# ---------------------------------------------------------------------------

def _call(stage="verify_draft", model="m", input_tok=100, output_tok=200,
          cost=0.001, latency=150.0, cache=False):
    return {
        "stage": stage, "model": model,
        "input_tokens": input_tok, "output_tokens": output_tok,
        "cost_usd": cost, "latency_ms": latency, "cache_hit": cache,
    }


def _rec(eid="e1", mid="M4", status="ok", usage=None, draft_usage=None):
    return {
        "example_id": eid,
        "method_id": mid,
        "status": status,
        "usage": usage or [],
        "draft_usage": draft_usage or [],
    }


# ---------------------------------------------------------------------------
# load_pricing
# ---------------------------------------------------------------------------

class TestLoadPricing:
    def test_loads_default_config(self):
        pricing = load_pricing()
        assert "mock-model-v1" in pricing
        assert pricing["mock-model-v1"]["input_per_1k_tokens"] == 0.0

    def test_returns_empty_dict_for_missing_file(self, tmp_path):
        result = load_pricing(tmp_path / "nonexistent.json")
        assert result == {}

    def test_loads_custom_json(self, tmp_path):
        cfg = {"models": {"test-model": {"input_per_1k_tokens": 0.01, "output_per_1k_tokens": 0.03}}}
        path = tmp_path / "pricing.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        pricing = load_pricing(path)
        assert pricing["test-model"]["input_per_1k_tokens"] == pytest.approx(0.01)

    def test_returns_empty_dict_for_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        assert load_pricing(path) == {}


# ---------------------------------------------------------------------------
# compute_cost_from_tokens
# ---------------------------------------------------------------------------

class TestComputeCostFromTokens:
    PRICING = {"gpt-4o": {"input_per_1k_tokens": 0.005, "output_per_1k_tokens": 0.015}}

    def test_basic(self):
        cost = compute_cost_from_tokens(1000, 500, "gpt-4o", self.PRICING)
        # 1 * 0.005 + 0.5 * 0.015 = 0.0125
        assert cost == pytest.approx(0.0125)

    def test_unknown_model_returns_none(self):
        assert compute_cost_from_tokens(100, 100, "unknown", self.PRICING) is None

    def test_none_tokens_returns_none(self):
        assert compute_cost_from_tokens(None, 100, "gpt-4o", self.PRICING) is None
        assert compute_cost_from_tokens(100, None, "gpt-4o", self.PRICING) is None

    def test_zero_tokens(self):
        cost = compute_cost_from_tokens(0, 0, "gpt-4o", self.PRICING)
        assert cost == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# aggregate_usage_by_method
# ---------------------------------------------------------------------------

class TestAggregateUsageByMethod:
    def test_basic_aggregation(self):
        records = [
            _rec("e1", "M4", usage=[_call(cost=0.001), _call(cost=0.002)]),
            _rec("e2", "M4", usage=[_call(cost=0.001)]),
        ]
        result = aggregate_usage_by_method(records)
        m4 = result["M4"]
        assert m4["n_records"] == 2
        assert m4["n_ok"] == 2
        assert m4["n_error"] == 0
        assert m4["llm_calls"] == 3
        assert m4["cost_usd_sum"] == pytest.approx(0.004)
        assert m4["n_calls_with_cost"] == 3

    def test_error_record_counts_in_denominator_but_not_usage(self):
        records = [
            _rec("e1", "M4", usage=[_call(cost=0.001)]),
            _rec("e2", "M4", status="error"),
        ]
        result = aggregate_usage_by_method(records)
        m4 = result["M4"]
        assert m4["n_records"] == 2
        assert m4["n_ok"] == 1
        assert m4["n_error"] == 1
        assert m4["llm_calls"] == 1
        assert m4["cost_usd_sum"] == pytest.approx(0.001)

    def test_per_example_cost_uses_n_ok_denominator(self):
        records = [
            _rec("e1", "M3", usage=[_call(cost=0.002)]),
            _rec("e2", "M3", status="error"),     # error: doesn't contribute cost
        ]
        result = aggregate_usage_by_method(records)
        # cost_usd_sum=0.002, n_ok=1 → per_example=0.002
        assert result["M3"]["per_example_cost_usd"] == pytest.approx(0.002)

    def test_multiple_methods(self):
        records = [
            _rec("e1", "M1", usage=[]),
            _rec("e1", "M5", usage=[_call(cost=0.005)]),
        ]
        result = aggregate_usage_by_method(records)
        assert result["M1"]["llm_calls"] == 0
        # M1 has ok records + zero calls → cost is 0.0 (known, not None)
        assert result["M1"]["cost_usd_sum"] == pytest.approx(0.0)
        assert result["M5"]["cost_usd_sum"] == pytest.approx(0.005)

    def test_none_cost_excluded_from_sum(self):
        records = [
            _rec("e1", "M4", usage=[_call(cost=0.001), _call(cost=None)]),
        ]
        result = aggregate_usage_by_method(records)
        m4 = result["M4"]
        assert m4["cost_usd_sum"] == pytest.approx(0.001)
        assert m4["n_calls_with_cost"] == 1
        assert m4["llm_calls"] == 2

    def test_all_none_cost_gives_none_sum(self):
        records = [_rec("e1", "M4", usage=[_call(cost=None)])]
        assert aggregate_usage_by_method(records)["M4"]["cost_usd_sum"] is None

    def test_latency_averaged_correctly(self):
        records = [
            _rec("e1", "M4", usage=[_call(latency=100.0), _call(latency=200.0)]),
        ]
        result = aggregate_usage_by_method(records)
        assert result["M4"]["mean_latency_ms"] == pytest.approx(150.0)

    def test_cache_hits_counted(self):
        records = [
            _rec("e1", "M4", usage=[_call(cache=True), _call(cache=False)]),
        ]
        assert aggregate_usage_by_method(records)["M4"]["cache_hits"] == 1

    def test_draft_usage_excluded_by_default(self):
        records = [
            _rec("e1", "M1",
                 usage=[],
                 draft_usage=[_call(cost=0.0001)]),
        ]
        result = aggregate_usage_by_method(records, include_draft_usage=False)
        # M1 has ok record + zero method calls → cost is 0.0, not the draft cost
        assert result["M1"]["cost_usd_sum"] == pytest.approx(0.0)

    def test_draft_usage_included_when_flag_set(self):
        records = [
            _rec("e1", "M1",
                 usage=[],
                 draft_usage=[_call(cost=0.0001)]),
        ]
        result = aggregate_usage_by_method(records, include_draft_usage=True)
        assert result["M1"]["cost_usd_sum"] == pytest.approx(0.0001)

    def test_pricing_fallback_fills_none_cost(self):
        pricing = {"my-model": {"input_per_1k_tokens": 0.01, "output_per_1k_tokens": 0.02}}
        records = [
            _rec("e1", "M4", usage=[_call(model="my-model", input_tok=1000, output_tok=500, cost=None)]),
        ]
        result = aggregate_usage_by_method(records, pricing=pricing)
        # 1000/1000*0.01 + 500/1000*0.02 = 0.01 + 0.01 = 0.02
        assert result["M4"]["cost_usd_sum"] == pytest.approx(0.02)

    def test_empty_records(self):
        assert aggregate_usage_by_method([]) == {}


# ---------------------------------------------------------------------------
# aggregate_draft_usage
# ---------------------------------------------------------------------------

class TestAggregateDraftUsage:
    def test_counts_each_example_once(self):
        # Both M1 and M4 for "e1" share the same draft — count only once.
        records = [
            _rec("e1", "M1", draft_usage=[_call(cost=0.001)]),
            _rec("e1", "M4", draft_usage=[_call(cost=0.001)]),
            _rec("e2", "M1", draft_usage=[_call(cost=0.001)]),
        ]
        result = aggregate_draft_usage(records)
        assert result["n_examples"] == 2
        assert result["cost_usd_sum"] == pytest.approx(0.002)

    def test_empty_returns_zeros(self):
        result = aggregate_draft_usage([])
        assert result["n_examples"] == 0
        assert result["llm_calls"] == 0

    def test_none_draft_usage_handled(self):
        records = [_rec("e1", "M1")]   # draft_usage defaults to []
        result = aggregate_draft_usage(records)
        assert result["llm_calls"] == 0


# ---------------------------------------------------------------------------
# estimate_budget
# ---------------------------------------------------------------------------

class TestEstimateBudget:
    def _trial(self):
        # 2 examples × 5 methods = 10 records; each ok record has 1 call @ 0.001
        records = []
        for eid in ["e1", "e2"]:
            for mid in ["M1", "M2", "M3", "M4", "M5"]:
                n_calls = {"M1": 0, "M2": 1, "M3": 2, "M4": 2, "M5": 3}[mid]
                usage = [_call(cost=0.001) for _ in range(n_calls)]
                records.append(_rec(
                    eid=eid, mid=mid,
                    usage=usage,
                    draft_usage=[_call(stage="generate_draft", cost=0.0005)],
                ))
        return records

    def test_scale_to_larger_run(self):
        result = estimate_budget(self._trial(), target_n_examples=100)
        assert result["trial_n_examples"] == 2
        assert result["target_n_examples"] == 100
        m4 = result["methods"]["M4"]
        # trial: 2 ok records × 2 calls × 0.001 = 0.004; per example = 0.002
        assert m4["per_example_cost_usd"] == pytest.approx(0.002)
        assert m4["estimated_cost_usd"] == pytest.approx(0.2)

    def test_m1_zero_method_calls(self):
        result = estimate_budget(self._trial(), target_n_examples=50)
        m1 = result["methods"]["M1"]
        assert m1["trial_calls"] == 0
        # M1 has ok records + zero calls → cost is 0.0 (known, not None)
        assert m1["per_example_cost_usd"] == pytest.approx(0.0)
        assert m1["estimated_cost_usd"] == pytest.approx(0.0)

    def test_draft_cost_estimated_separately(self):
        result = estimate_budget(self._trial(), target_n_examples=10)
        draft = result["draft"]
        # trial: 2 examples × 1 draft call × 0.0005 = 0.001; per example = 0.0005
        assert draft["per_example_cost_usd"] == pytest.approx(0.0005)
        assert draft["estimated_cost_usd"] == pytest.approx(0.005)

    def test_total_is_none_when_calls_exist_but_no_pricing(self):
        # Records where calls have cost=None AND no pricing provided → total is None
        records = [
            _rec("e1", "M4", usage=[_call(cost=None)]),
            _rec("e2", "M4", usage=[_call(cost=None)]),
        ]
        result = estimate_budget(records, target_n_examples=100)
        assert result["total_estimated_cost_usd"] is None

    def test_total_not_none_when_m1_has_zero_calls(self):
        # M1 has ok records + zero calls → 0.0 cost; total must not be None
        result = estimate_budget(self._trial(), target_n_examples=100)
        assert result["total_estimated_cost_usd"] is not None

    def test_total_sums_when_all_costs_known(self):
        # Only M4 records, one call each, cost=0.001
        records = [
            _rec("e1", "M4", usage=[_call(cost=0.001)], draft_usage=[_call(cost=0.0005)]),
            _rec("e2", "M4", usage=[_call(cost=0.001)], draft_usage=[_call(cost=0.0005)]),
        ]
        result = estimate_budget(records, target_n_examples=10)
        # method: 0.001 per example × 10 = 0.01
        # draft: 0.0005 per example × 10 = 0.005
        # total = 0.015
        assert result["total_estimated_cost_usd"] == pytest.approx(0.015)

    def test_estimated_calls_rounded(self):
        records = [
            _rec("e1", "M5", usage=[_call(), _call(), _call()]),
        ]
        result = estimate_budget(records, target_n_examples=10)
        # 3 calls per example × 10 = 30
        assert result["methods"]["M5"]["estimated_calls"] == 30

    def test_empty_trial_gives_zero_examples(self):
        result = estimate_budget([], target_n_examples=100)
        assert result["trial_n_examples"] == 0
        assert result["total_estimated_cost_usd"] is None


# ---------------------------------------------------------------------------
# Error cost estimation in estimate_budget
# ---------------------------------------------------------------------------

class TestEstimateBudgetWithErrors:
    def test_estimated_error_cost_extrapolated(self):
        records = [
            _rec("e1", "M4", usage=[_call(cost=0.002)]),
            _rec("e2", "M4", usage=[_call(cost=0.002)]),
            _rec("e3", "M4", status="error", usage=[_call(cost=0.001)]),
        ]
        result = estimate_budget(records, target_n_examples=30)
        m4 = result["methods"]["M4"]
        # ok: 2 records, cost=0.004, per_ok=0.002 → est_ok = 0.002*30 = 0.06
        assert m4["estimated_ok_cost"] == pytest.approx(0.06)
        # error: 1 error in 3 records, error_cost=0.001 → per_ex=0.001/3
        # est_error = (0.001/3) * 30 = 0.01
        assert m4["estimated_error_cost"] == pytest.approx(0.01)
        assert m4["estimated_cost_usd"] == pytest.approx(0.07)

    def test_no_errors_gives_zero_estimated_error_cost(self):
        records = [_rec("e1", "M4", usage=[_call(cost=0.002)])]
        result = estimate_budget(records, target_n_examples=10)
        m4 = result["methods"]["M4"]
        assert m4["estimated_error_cost"] == pytest.approx(0.0)
        assert m4["estimated_cost_usd"] == pytest.approx(0.02)

    def test_grand_total_includes_error_cost(self):
        records = [
            _rec("e1", "M4", usage=[_call(cost=0.002)],
                 draft_usage=[_call(stage="generate_draft", cost=0.001)]),
            _rec("e2", "M4", status="error", usage=[_call(cost=0.001)]),
        ]
        result = estimate_budget(records, target_n_examples=10)
        # ok: n_ok=1, per_ok=0.002, est_ok=0.02
        # error: n_records=2, error_cost=0.001, per_ex=0.0005, est_error=0.005
        # draft: n_examples=2, draft_cost=0.001, per_ex=0.0005, est_draft=0.005
        # total = 0.02 + 0.005 + 0.005 = 0.03
        assert result["total_estimated_cost_usd"] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# Issue 2: M1 zero-call cost is 0.0, not None
# ---------------------------------------------------------------------------

class TestZeroCallCost:
    def test_ok_records_zero_calls_gives_zero_cost(self):
        records = [
            _rec("e1", "M1", usage=[]),
            _rec("e2", "M1", usage=[]),
        ]
        result = aggregate_usage_by_method(records)
        m1 = result["M1"]
        assert m1["n_ok"] == 2
        assert m1["llm_calls"] == 0
        assert m1["cost_usd_sum"] == pytest.approx(0.0), (
            "Method with ok records and zero calls should have cost 0.0, not None"
        )

    def test_per_example_cost_zero_when_no_calls(self):
        records = [_rec("e1", "M1", usage=[])]
        result = aggregate_usage_by_method(records)
        assert result["M1"]["per_example_cost_usd"] == pytest.approx(0.0)

    def test_only_error_records_still_none(self):
        records = [_rec("e1", "M1", status="error", usage=[])]
        result = aggregate_usage_by_method(records)
        # No ok records → cost unknown → None
        assert result["M1"]["cost_usd_sum"] is None


# ---------------------------------------------------------------------------
# Issue 4: error records preserve usage; cost tracked separately
# ---------------------------------------------------------------------------

class TestErrorCostTracking:
    def test_error_cost_usd_sum_in_output(self):
        records = [_rec("e1", "M4", status="error",
                        usage=[_call(cost=0.005)])]
        result = aggregate_usage_by_method(records)
        assert "error_cost_usd_sum" in result["M4"], (
            "error_cost_usd_sum key must be present in output"
        )

    def test_error_record_cost_tracked_separately(self):
        records = [
            _rec("e1", "M4", usage=[_call(cost=0.001)]),
            _rec("e2", "M4", status="error", usage=[_call(cost=0.005)]),
        ]
        result = aggregate_usage_by_method(records)
        m4 = result["M4"]
        assert m4["cost_usd_sum"] == pytest.approx(0.001), "ok cost only"
        assert m4["error_cost_usd_sum"] == pytest.approx(0.005), "error cost only"

    def test_ok_cost_excludes_error_records(self):
        records = [
            _rec("e1", "M2", usage=[_call(cost=0.002)]),
            _rec("e2", "M2", status="error", usage=[_call(cost=0.008)]),
        ]
        result = aggregate_usage_by_method(records)
        assert result["M2"]["cost_usd_sum"] == pytest.approx(0.002)

    def test_no_error_usage_gives_none_error_cost(self):
        records = [_rec("e1", "M4", usage=[_call(cost=0.001)])]
        result = aggregate_usage_by_method(records)
        assert result["M4"]["error_cost_usd_sum"] is None


# ---------------------------------------------------------------------------
# Issue 9: cache_hit=True costs excluded from cost_usd_sum
# ---------------------------------------------------------------------------

class TestCacheHitCost:
    def test_cache_hit_cost_excluded_from_paid_cost(self):
        records = [
            _rec("e1", "M4", usage=[_call(cost=0.002, cache=False),
                                    _call(cost=0.001, cache=True)]),
        ]
        result = aggregate_usage_by_method(records)
        m4 = result["M4"]
        # Only non-cached call's cost counts as paid
        assert m4["cost_usd_sum"] == pytest.approx(0.002)

    def test_cached_cost_sum_in_output(self):
        records = [_rec("e1", "M4", usage=[_call(cost=0.001, cache=True)])]
        result = aggregate_usage_by_method(records)
        assert "cached_cost_sum" in result["M4"], (
            "cached_cost_sum key must be in output"
        )

    def test_cached_cost_sum_value(self):
        records = [
            _rec("e1", "M4", usage=[_call(cost=0.001, cache=True),
                                    _call(cost=0.002, cache=True)]),
        ]
        result = aggregate_usage_by_method(records)
        assert result["M4"]["cached_cost_sum"] == pytest.approx(0.003)

    def test_all_cache_hits_gives_zero_paid_cost(self):
        # All calls are cache hits → paid cost should be 0.0 (not None)
        records = [
            _rec("e1", "M4", usage=[_call(cost=0.001, cache=True)]),
            _rec("e2", "M4", usage=[_call(cost=0.001, cache=True)]),
        ]
        result = aggregate_usage_by_method(records)
        m4 = result["M4"]
        assert m4["cost_usd_sum"] == pytest.approx(0.0), (
            "All calls cached → paid cost is 0.0 (known zero, not None)"
        )

    def test_non_cached_calls_counted_normally(self):
        records = [_rec("e1", "M4", usage=[_call(cost=0.003, cache=False)])]
        result = aggregate_usage_by_method(records)
        assert result["M4"]["cost_usd_sum"] == pytest.approx(0.003)
        assert result["M4"]["cached_cost_sum"] is None
