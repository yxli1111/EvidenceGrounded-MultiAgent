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
        # "paris paris" vs "paris": Counter intersection = {"paris":1}
        # pred_tokens=["paris","paris"] gold_tokens=["paris"]
        # precision=1/2, recall=1/1, f1=2/3
        f1 = token_f1("paris paris", "paris")
        assert f1 == pytest.approx(2 / 3, abs=1e-6)


# ---------------------------------------------------------------------------
# HotpotQA yes/no/noanswer special case (Issue 5)
# ---------------------------------------------------------------------------

class TestYesNoSpecialCase:
    def test_yes_yes_f1_is_one(self):
        assert token_f1("yes", "yes") == pytest.approx(1.0)

    def test_yes_no_f1_is_zero(self):
        assert token_f1("yes", "no") == pytest.approx(0.0)

    def test_no_yes_f1_is_zero(self):
        assert token_f1("no", "yes") == pytest.approx(0.0)

    def test_noanswer_noanswer_f1_is_one(self):
        assert token_f1("noanswer", "noanswer") == pytest.approx(1.0)

    def test_gold_yes_pred_other_gets_zero_not_partial(self):
        # Without the special case: token_f1("yes please", "yes")
        # pred=["yes","please"], gold=["yes"] → precision=1/2, recall=1, f1=2/3
        # With special case: gold="yes" ∈ {yes,no,noanswer} → 0.0 (no exact match)
        assert token_f1("yes please", "yes") == pytest.approx(0.0)

    def test_gold_no_pred_mixed_gets_zero(self):
        # Without special case: token_f1("yes no", "no")
        # pred=["yes","no"], gold=["no"] → precision=1/2, recall=1, f1=2/3
        # With special case: 0.0
        assert token_f1("yes no", "no") == pytest.approx(0.0)

    def test_yes_no_em_still_zero(self):
        assert exact_match("yes", "no") == pytest.approx(0.0)

    def test_yes_yes_em_is_one(self):
        assert exact_match("yes", "yes") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# None final_answer handling (Issue 5)
# ---------------------------------------------------------------------------

class TestNoneFinalAnswer:
    def test_none_prediction_exact_match_is_zero(self):
        assert exact_match(None, "Paris") == pytest.approx(0.0)

    def test_none_gold_exact_match_is_zero(self):
        assert exact_match("Paris", None) == pytest.approx(0.0)

    def test_none_both_exact_match_is_one(self):
        # Both None → both normalize to "" → equal
        assert exact_match(None, None) == pytest.approx(1.0)

    def test_none_prediction_f1_is_zero(self):
        assert token_f1(None, "Paris") == pytest.approx(0.0)

    def test_compute_batch_none_final_answer_no_crash(self):
        records = [{"example_id": "e1", "method_id": "M1",
                    "status": "ok", "final_answer": None}]
        gold = {"e1": {"gold_answer": "Paris"}}
        result = compute_em_f1_batch(records, gold)
        assert result["M1"]["em"] == pytest.approx(0.0)


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


# ---------------------------------------------------------------------------
# export_per_example_csv
# ---------------------------------------------------------------------------

class TestExportPerExampleCsv:
    def _make_record(self, eid, mid, status, pred, short, qt=None):
        return {
            "example_id": eid,
            "method_id": mid,
            "status": status,
            "final_answer": pred,
            "short_answer": short,
            "question_type": qt,
        }

    def test_csv_file_created(self, tmp_path):
        from src.evaluation.answer_metrics import export_per_example_csv
        out = str(tmp_path / "out.csv")
        records = [self._make_record("e1", "M1", "ok", "Paris", "Paris", "factoid")]
        gold = {"e1": {"gold_answer": "Paris"}}
        path = export_per_example_csv(records, gold, output_path=out)
        import os
        assert os.path.exists(path)

    def test_csv_columns_correct(self, tmp_path):
        from src.evaluation.answer_metrics import export_per_example_csv
        import csv
        out = str(tmp_path / "cols.csv")
        records = [self._make_record("e1", "M1", "ok", "Paris", "Paris")]
        gold = {"e1": {"gold_answer": "Paris"}}
        # Use preliminary=False to get a clean CSV without comment header
        export_per_example_csv(records, gold, output_path=out, preliminary=False)
        with open(out, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        assert header == ["example_id", "method_id", "status", "em", "f1",
                          "draft_em", "question_type"]

    def test_em_f1_values_correct(self, tmp_path):
        from src.evaluation.answer_metrics import export_per_example_csv
        import csv
        out = str(tmp_path / "vals.csv")
        records = [self._make_record("e1", "M1", "ok", "Paris", "London")]
        gold = {"e1": {"gold_answer": "Paris"}}
        export_per_example_csv(records, gold, output_path=out, preliminary=False)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert float(rows[0]["em"]) == pytest.approx(1.0)
        assert float(rows[0]["draft_em"]) == pytest.approx(0.0)  # London != Paris

    def test_error_status_gives_zero_scores(self, tmp_path):
        from src.evaluation.answer_metrics import export_per_example_csv
        import csv
        out = str(tmp_path / "err.csv")
        records = [self._make_record("e1", "M1", "error", None, None)]
        gold = {"e1": {"gold_answer": "Paris"}}
        export_per_example_csv(records, gold, output_path=out, preliminary=False)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["em"]) == pytest.approx(0.0)
        assert float(rows[0]["f1"]) == pytest.approx(0.0)

    def test_question_type_in_output(self, tmp_path):
        from src.evaluation.answer_metrics import export_per_example_csv
        import csv
        out = str(tmp_path / "qt.csv")
        records = [self._make_record("e1", "M1", "ok", "yes", "yes", "yesno")]
        gold = {"e1": {"gold_answer": "yes"}}
        export_per_example_csv(records, gold, output_path=out, preliminary=False)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["question_type"] == "yesno"

    def test_parent_dirs_created(self, tmp_path):
        from src.evaluation.answer_metrics import export_per_example_csv
        out = str(tmp_path / "deep" / "nested" / "out.csv")
        export_per_example_csv([], {}, output_path=out)
        import os
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# export_per_example_csv — new behaviour (Items 2a-2c)
# ---------------------------------------------------------------------------

class TestExportPerExampleCsvNewFormat:
    def _rec(self, eid, mid, pred, short, status="ok", attempt=1, qt=None):
        return {
            "example_id": eid,
            "method_id": mid,
            "status": status,
            "final_answer": pred,
            "short_answer": short,
            "attempt": attempt,
            "question_type": qt,
        }

    def test_no_comment_line_in_csv(self, tmp_path):
        """CSV must not start with '#'; pandas.read_csv must read it directly."""
        from src.evaluation.answer_metrics import export_per_example_csv
        out = str(tmp_path / "out.csv")
        records = [self._rec("e1", "M1", "Paris", "Paris")]
        gold = {"e1": {"gold_answer": "Paris"}}
        export_per_example_csv(records, gold, output_path=out, preliminary=True)
        with open(out, encoding="utf-8") as f:
            first = f.readline().strip()
        assert not first.startswith("#"), "CSV must not start with a comment line"
        assert "example_id" in first

    def test_metadata_json_written_alongside_csv(self, tmp_path):
        """metadata.json is created in the same directory as the CSV."""
        import json
        from src.evaluation.answer_metrics import export_per_example_csv
        out = str(tmp_path / "out.csv")
        records = [self._rec("e1", "M1", "Paris", "Paris")]
        gold = {"e1": {"gold_answer": "Paris"}}
        export_per_example_csv(records, gold, output_path=out, preliminary=True)
        meta = tmp_path / "metadata.json"
        assert meta.exists(), "metadata.json must be created alongside CSV"
        data = json.loads(meta.read_text())
        assert data["preliminary"] is True
        assert "n" in data

    def test_metadata_json_n_reflects_deduped_count(self, tmp_path):
        """metadata.json n equals the number of deduped rows written."""
        import json
        from src.evaluation.answer_metrics import export_per_example_csv
        out = str(tmp_path / "out.csv")
        records = [
            self._rec("e1", "M1", "Wrong", "Wrong", attempt=1),
            self._rec("e1", "M1", "Paris", "Paris", attempt=2),  # duplicate
        ]
        gold = {"e1": {"gold_answer": "Paris"}}
        export_per_example_csv(records, gold, output_path=out)
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta["n"] == 1  # deduped

    def test_dedup_keeps_highest_attempt_in_csv(self, tmp_path):
        """Two attempts for same (eid, mid); only highest attempt row in CSV."""
        import csv
        from src.evaluation.answer_metrics import export_per_example_csv
        out = str(tmp_path / "out.csv")
        records = [
            self._rec("e1", "M1", "Wrong", "Wrong", attempt=1),
            self._rec("e1", "M1", "Paris", "Paris", attempt=2),
        ]
        gold = {"e1": {"gold_answer": "Paris"}}
        export_per_example_csv(records, gold, output_path=out, preliminary=False)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1, "deduped: only one row"
        assert float(rows[0]["em"]) == pytest.approx(1.0), "highest attempt (Paris)"

    def test_pandas_reads_csv_without_skiprows(self, tmp_path):
        """pandas.read_csv must work without comment= or skiprows= argument."""
        from src.evaluation.answer_metrics import export_per_example_csv
        out = str(tmp_path / "out.csv")
        records = [self._rec("e1", "M1", "Paris", "Paris")]
        gold = {"e1": {"gold_answer": "Paris"}}
        export_per_example_csv(records, gold, output_path=out, preliminary=True)
        try:
            import pandas as pd
            df = pd.read_csv(out)
            assert "example_id" in df.columns
        except ImportError:
            pytest.skip("pandas not installed")

    def test_metadata_omitted_metrics_field(self, tmp_path):
        """omitted_metrics list is written to metadata.json when provided."""
        import json
        from src.evaluation.answer_metrics import export_per_example_csv
        out = str(tmp_path / "out.csv")
        export_per_example_csv([], {}, output_path=out,
                                omitted_metrics=["retrieval", "cost"])
        meta = json.loads((tmp_path / "metadata.json").read_text())
        assert meta.get("omitted_metrics") == ["retrieval", "cost"]
