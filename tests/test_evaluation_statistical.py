"""Unit tests for src/evaluation/statistical.py.

All inputs are synthetic; no real data or API calls used.
Bootstrap results use a fixed seed for reproducibility.
"""

import pytest

from src.evaluation.statistical import compare_methods, paired_bootstrap_ci


# ---------------------------------------------------------------------------
# paired_bootstrap_ci
# ---------------------------------------------------------------------------

class TestPairedBootstrapCI:
    def test_zero_difference(self):
        a = [1.0] * 10
        b = [1.0] * 10
        result = paired_bootstrap_ci(a, b, n_resamples=500, seed=0)
        assert result["delta_obs"] == pytest.approx(0.0)
        assert result["ci_low"] == pytest.approx(0.0)
        assert result["ci_high"] == pytest.approx(0.0)

    def test_positive_difference(self):
        a = [1.0] * 10
        b = [0.0] * 10
        result = paired_bootstrap_ci(a, b, n_resamples=1000, seed=42)
        assert result["delta_obs"] == pytest.approx(1.0)
        assert result["ci_low"] > 0.5
        assert result["ci_high"] == pytest.approx(1.0)
        assert result["p_value"] == pytest.approx(0.0)

    def test_negative_difference(self):
        a = [0.0] * 10
        b = [1.0] * 10
        result = paired_bootstrap_ci(a, b, n_resamples=1000, seed=42)
        assert result["delta_obs"] == pytest.approx(-1.0)
        assert result["ci_high"] < -0.5
        assert result["p_value"] == pytest.approx(1.0)

    def test_returns_correct_keys(self):
        result = paired_bootstrap_ci([0.5, 0.6], [0.4, 0.5], n_resamples=100, seed=1)
        assert set(result.keys()) == {
            "n", "delta_obs", "ci_low", "ci_high",
            "confidence", "p_value", "n_resamples",
        }

    def test_n_is_correct(self):
        result = paired_bootstrap_ci([0.1, 0.2, 0.3], [0.1, 0.1, 0.1], n_resamples=100, seed=1)
        assert result["n"] == 3

    def test_confidence_stored(self):
        result = paired_bootstrap_ci([1.0], [0.0], n_resamples=100, seed=1, confidence=0.90)
        assert result["confidence"] == pytest.approx(0.90)

    def test_n_resamples_stored(self):
        result = paired_bootstrap_ci([1.0], [0.0], n_resamples=200, seed=1)
        assert result["n_resamples"] == 200

    def test_ci_ordered(self):
        rng_scores = [float(i % 2) for i in range(20)]
        result = paired_bootstrap_ci(rng_scores, [0.5] * 20, n_resamples=500, seed=7)
        assert result["ci_low"] <= result["ci_high"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            paired_bootstrap_ci([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            paired_bootstrap_ci([1.0, 2.0], [1.0])

    def test_reproducible_with_seed(self):
        a = [float(i % 2) for i in range(30)]
        b = [float((i + 1) % 2) for i in range(30)]
        r1 = paired_bootstrap_ci(a, b, n_resamples=500, seed=99)
        r2 = paired_bootstrap_ci(a, b, n_resamples=500, seed=99)
        assert r1["ci_low"] == pytest.approx(r2["ci_low"])
        assert r1["ci_high"] == pytest.approx(r2["ci_high"])
        assert r1["p_value"] == pytest.approx(r2["p_value"])

    def test_different_seeds_may_differ(self):
        a = [float(i % 3) / 2 for i in range(20)]
        b = [0.3] * 20
        r1 = paired_bootstrap_ci(a, b, n_resamples=200, seed=1)
        r2 = paired_bootstrap_ci(a, b, n_resamples=200, seed=2)
        # Won't always differ, but delta_obs is deterministic
        assert r1["delta_obs"] == pytest.approx(r2["delta_obs"])

    def test_single_example(self):
        result = paired_bootstrap_ci([1.0], [0.0], n_resamples=100, seed=0)
        assert result["n"] == 1
        assert result["delta_obs"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compare_methods
# ---------------------------------------------------------------------------

def _make_record(eid, mid, answer, status="ok"):
    return {
        "example_id": eid,
        "method_id": mid,
        "status": status,
        "final_answer": answer,
        "usage": [],
        "draft_usage": [],
    }


GOLD = {
    "e1": {"gold_answer": "Paris"},
    "e2": {"gold_answer": "Berlin"},
    "e3": {"gold_answer": "Rome"},
}

RECORDS_EM = [
    # M4 correct on e1, e2; M5 correct on e1, e2, e3
    _make_record("e1", "M4", "Paris"),
    _make_record("e2", "M4", "Berlin"),
    _make_record("e3", "M4", "wrong"),
    _make_record("e1", "M5", "Paris"),
    _make_record("e2", "M5", "Berlin"),
    _make_record("e3", "M5", "Rome"),
]


class TestCompareMethods:
    def test_delta_obs_correct(self):
        result = compare_methods(RECORDS_EM, GOLD, "M4", "M5",
                                 metric="em", n_resamples=500, seed=0)
        # M4 EM = 2/3, M5 EM = 3/3; delta = -1/3
        assert result["delta_obs"] == pytest.approx(-1 / 3, abs=1e-9)

    def test_mean_a_b(self):
        result = compare_methods(RECORDS_EM, GOLD, "M4", "M5",
                                 metric="em", n_resamples=500, seed=0)
        assert result["mean_a"] == pytest.approx(2 / 3, abs=1e-9)
        assert result["mean_b"] == pytest.approx(1.0)

    def test_extra_keys_present(self):
        result = compare_methods(RECORDS_EM, GOLD, "M4", "M5",
                                 n_resamples=100, seed=0)
        assert result["method_a"] == "M4"
        assert result["method_b"] == "M5"
        assert result["metric"] == "em"

    def test_no_paired_examples_raises(self):
        # Neither M4 nor M5 has any records → truly no examples to compare
        records = [_make_record("e1", "M3", "Paris")]
        with pytest.raises(ValueError, match="No paired"):
            compare_methods(records, GOLD, "M4", "M5", n_resamples=50, seed=0)

    def test_f1_metric(self):
        result = compare_methods(RECORDS_EM, GOLD, "M4", "M5",
                                 metric="f1", n_resamples=100, seed=0)
        assert result["metric"] == "f1"
        assert 0.0 <= result["mean_a"] <= 1.0
        assert 0.0 <= result["mean_b"] <= 1.0


# ---------------------------------------------------------------------------
# Issue 3: failed records count as 0; n_failed_a/b reported
# ---------------------------------------------------------------------------

class TestCompareMethodsFailedRecords:
    def test_error_record_counted_as_zero_not_dropped(self):
        # M5 has error for e3 → must be included with score 0.0, not dropped
        # RECORDS_EM: M4 e3=wrong(0), M5 e3=Rome(1).
        # Add M5 error for e3 to override:
        records = [
            _make_record("e1", "M4", "Paris"),
            _make_record("e2", "M4", "Berlin"),
            _make_record("e3", "M4", "wrong"),
            _make_record("e1", "M5", "Paris"),
            _make_record("e2", "M5", "Berlin"),
            _make_record("e3", "M5", "Rome", status="error"),
        ]
        result = compare_methods(records, GOLD, "M4", "M5",
                                 metric="em", n_resamples=200, seed=0)
        # n must be 3 (e3 included despite M5 error)
        assert result["n"] == 3
        # M5 scores: e1=1, e2=1, e3=0 (error) → mean_b = 2/3
        assert result["mean_b"] == pytest.approx(2 / 3, abs=1e-9)

    def test_missing_method_b_record_counts_as_zero(self):
        # M5 has NO record for e3 → e3 should be included with M5 score=0
        records = [
            _make_record("e1", "M4", "Paris"),
            _make_record("e2", "M4", "Berlin"),
            _make_record("e3", "M4", "wrong"),
            _make_record("e1", "M5", "Paris"),
            _make_record("e2", "M5", "Berlin"),
            # no M5 record for e3
        ]
        result = compare_methods(records, GOLD, "M4", "M5",
                                 metric="em", n_resamples=200, seed=0)
        assert result["n"] == 3, "e3 must be included even though M5 has no record"
        # M5: e1=1, e2=1, e3=0 (missing) → mean_b = 2/3
        assert result["mean_b"] == pytest.approx(2 / 3, abs=1e-9)

    def test_n_failed_a_and_n_failed_b_in_result(self):
        result = compare_methods(RECORDS_EM, GOLD, "M4", "M5",
                                 n_resamples=100, seed=0)
        assert "n_failed_a" in result, "n_failed_a must be in result"
        assert "n_failed_b" in result, "n_failed_b must be in result"

    def test_n_failed_b_counts_error_records(self):
        records = [
            _make_record("e1", "M4", "Paris"),
            _make_record("e2", "M4", "Berlin"),
            _make_record("e3", "M4", "wrong"),
            _make_record("e1", "M5", "Paris"),
            _make_record("e2", "M5", "Rome", status="error"),
            _make_record("e3", "M5", "Rome", status="error"),
        ]
        result = compare_methods(records, GOLD, "M4", "M5",
                                 n_resamples=100, seed=0)
        assert result["n_failed_b"] == 2

    def test_excludes_only_example_missing_from_both(self):
        # e1 in M4 only, e2 in M5 only, e3 in both
        records = [
            _make_record("e1", "M4", "Paris"),
            _make_record("e2", "M5", "Berlin"),
            _make_record("e3", "M4", "wrong"),
            _make_record("e3", "M5", "Rome"),
        ]
        result = compare_methods(records, GOLD, "M4", "M5",
                                 n_resamples=100, seed=0)
        # All three examples appear in at least one method → n=3
        assert result["n"] == 3
