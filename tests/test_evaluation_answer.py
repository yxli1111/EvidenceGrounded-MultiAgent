"""Unit tests for src/evaluation/answer_metrics.py.

All expected values are hand-derived from the HotpotQA normalization rules.
No external data or API calls are used.
"""

import pytest
from src.evaluation.answer_metrics import (
    normalize_answer,
    exact_match,
    token_f1,
    compute_em_f1_batch,
)


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------

class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Paris") == "paris"

    def test_strips_articles(self):
        assert normalize_answer("the cat") == "cat"
        assert normalize_answer("a dog") == "dog"
        assert normalize_answer("an apple") == "apple"

    def test_strips_punctuation(self):
        assert normalize_answer("Paris, France") == "paris france"

    def test_collapses_whitespace(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_combined(self):
        assert normalize_answer("The Eiffel Tower, Paris.") == "eiffel tower paris"

    def test_empty_string(self):
        assert normalize_answer("") == ""


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_identical(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        assert exact_match("paris", "Paris") == 1.0

    def test_article_stripped(self):
        assert exact_match("the Eiffel Tower", "Eiffel Tower") == 1.0

    def test_punctuation_stripped(self):
        assert exact_match("Paris.", "Paris") == 1.0

    def test_different(self):
        assert exact_match("London", "Paris") == 0.0

    def test_partial_no_match(self):
        assert exact_match("Paris, France", "Paris") == 0.0

    def test_yes_no(self):
        assert exact_match("yes", "no") == 0.0
        assert exact_match("yes", "yes") == 1.0

    def test_empty_both(self):
        assert exact_match("", "") == 1.0

    def test_empty_prediction(self):
        assert exact_match("", "Paris") == 0.0


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------

class TestTokenF1:
    def test_identical(self):
        assert token_f1("Paris", "Paris") == pytest.approx(1.0)

    def test_full_overlap_different_order(self):
        # "the capital paris" vs "paris capital" after normalization:
        # pred_tokens = ["capital", "paris"]  (article stripped)
        # gold_tokens = ["paris", "capital"]  (article stripped)
        f1 = token_f1("the capital Paris", "Paris capital")
        assert f1 == pytest.approx(1.0)

    def test_partial_overlap(self):
        # pred: ["paris", "france"]  gold: ["paris"]
        # precision = 1/2, recall = 1/1, f1 = 2*(0.5*1)/(0.5+1) = 0.667
        f1 = token_f1("Paris France", "Paris")
        assert f1 == pytest.approx(2 / 3, abs=1e-6)

    def test_no_overlap(self):
        assert token_f1("London", "Paris") == pytest.approx(0.0)

    def test_yes_no(self):
        assert token_f1("yes", "no") == pytest.approx(0.0)

    def test_empty_both(self):
        assert token_f1("", "") == pytest.approx(1.0)

    def test_empty_prediction(self):
        assert token_f1("", "Paris") == pytest.approx(0.0)

    def test_empty_gold(self):
        assert token_f1("Paris", "") == pytest.approx(0.0)

    def test_duplicate_tokens(self):
        # "yes yes" vs "yes": common Counter intersection = {"yes":1}
        # pred_tokens=["yes","yes"] gold_tokens=["yes"]
        # precision=1/2, recall=1/1, f1=2/3
        f1 = token_f1("yes yes", "yes")
        assert f1 == pytest.approx(2 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_em_f1_batch
# ---------------------------------------------------------------------------

class TestComputeEmF1Batch:
    def _make_records(self, rows):
        return [
            {
                "example_id": r["eid"],
                "method_id": r["mid"],
                "status": r.get("status", "ok"),
                "final_answer": r.get("pred", ""),
            }
            for r in rows
        ]

    def _make_gold(self, mapping):
        return {eid: {"gold_answer": ans} for eid, ans in mapping.items()}

    def test_perfect_batch(self):
        records = self._make_records([
            {"eid": "e1", "mid": "M1", "pred": "Paris"},
            {"eid": "e2", "mid": "M1", "pred": "Berlin"},
        ])
        gold = self._make_gold({"e1": "Paris", "e2": "Berlin"})
        result = compute_em_f1_batch(records, gold)
        assert result["M1"]["em"] == pytest.approx(1.0)
        assert result["M1"]["f1"] == pytest.approx(1.0)
        assert result["M1"]["n"] == 2
        assert result["M1"]["n_ok"] == 2

    def test_error_record_counts_in_denominator(self):
        records = self._make_records([
            {"eid": "e1", "mid": "M1", "pred": "Paris"},
            {"eid": "e2", "mid": "M1", "status": "error", "pred": ""},
        ])
        gold = self._make_gold({"e1": "Paris", "e2": "Berlin"})
        result = compute_em_f1_batch(records, gold)
        assert result["M1"]["n"] == 2
        assert result["M1"]["n_error"] == 1
        # em_sum=1.0, denominator=2 -> 0.5
        assert result["M1"]["em"] == pytest.approx(0.5)

    def test_multiple_methods(self):
        records = self._make_records([
            {"eid": "e1", "mid": "M1", "pred": "Paris"},
            {"eid": "e1", "mid": "M2", "pred": "London"},
        ])
        gold = self._make_gold({"e1": "Paris"})
        result = compute_em_f1_batch(records, gold)
        assert result["M1"]["em"] == pytest.approx(1.0)
        assert result["M2"]["em"] == pytest.approx(0.0)

    def test_empty_records(self):
        result = compute_em_f1_batch([], {})
        assert result == {}

    def test_skipped_status_counts_as_error(self):
        records = self._make_records([
            {"eid": "e1", "mid": "M1", "status": "skipped", "pred": "Paris"},
        ])
        gold = self._make_gold({"e1": "Paris"})
        result = compute_em_f1_batch(records, gold)
        assert result["M1"]["em"] == pytest.approx(0.0)
        assert result["M1"]["n_error"] == 1
