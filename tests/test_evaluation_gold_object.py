"""Tests for QAGold object support in evaluation functions (item 2).

All evaluation functions must accept gold_lookup values that are either:
  - dict: {"gold_answer": str, "gold_supporting_facts": list}  (B's original format)
  - QAGold object with .gold_answer and .gold_supporting_facts attributes

Duck-type tests run unconditionally.
Tests that import contracts.schemas.QAGold are gated with pytest.importorskip.
"""

import pytest


# ---------------------------------------------------------------------------
# Helper: duck-type QAGold (no contracts dependency)
# ---------------------------------------------------------------------------

class _DuckQAGold:
    """Plain object that mimics contracts.schemas.QAGold attribute access."""
    def __init__(self, gold_answer, gold_supporting_facts):
        self.gold_answer = gold_answer
        self.gold_supporting_facts = gold_supporting_facts


def _make_record(example_id, method_id, status="ok", final_answer="Paris",
                 short_answer="Paris", evidence=None):
    return {
        "example_id": example_id,
        "method_id": method_id,
        "status": status,
        "final_answer": final_answer,
        "short_answer": short_answer,
        "evidence": evidence or [],
    }


# ---------------------------------------------------------------------------
# compute_em_f1_batch — QAGold object gold_lookup
# ---------------------------------------------------------------------------

class TestComputeEmF1BatchWithObjects:

    def test_gold_object_gives_same_result_as_dict(self):
        from src.evaluation.answer_metrics import compute_em_f1_batch

        records = [_make_record("e1", "M1", final_answer="Paris")]

        dict_lookup = {"e1": {"gold_answer": "Paris"}}
        obj_lookup  = {"e1": _DuckQAGold("Paris", [])}

        r_dict = compute_em_f1_batch(records, dict_lookup)
        r_obj  = compute_em_f1_batch(records, obj_lookup)

        assert r_obj["M1"]["em"] == r_dict["M1"]["em"]
        assert r_obj["M1"]["f1"] == r_dict["M1"]["f1"]

    def test_gold_object_wrong_answer_gives_zero(self):
        from src.evaluation.answer_metrics import compute_em_f1_batch

        records = [_make_record("e1", "M1", final_answer="Berlin")]
        lookup  = {"e1": _DuckQAGold("Paris", [])}
        result  = compute_em_f1_batch(records, lookup)
        assert result["M1"]["em"] == pytest.approx(0.0)

    def test_missing_gold_entry_scores_zero(self):
        from src.evaluation.answer_metrics import compute_em_f1_batch

        records = [_make_record("e_unknown", "M1", final_answer="Paris")]
        lookup  = {"e1": _DuckQAGold("Paris", [])}
        result  = compute_em_f1_batch(records, lookup)
        assert result["M1"]["em"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# export_per_example_csv — QAGold object gold_lookup
# ---------------------------------------------------------------------------

class TestExportCsvWithObjects:

    def test_csv_written_with_object_gold(self, tmp_path):
        from src.evaluation.answer_metrics import export_per_example_csv

        records = [_make_record("e1", "M1", final_answer="Paris")]
        lookup  = {"e1": _DuckQAGold("Paris", [])}
        path    = export_per_example_csv(records, lookup, str(tmp_path / "out.csv"))
        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert float(rows[0]["em"]) == pytest.approx(1.0)

    def test_csv_wrong_answer_with_object_gold(self, tmp_path):
        from src.evaluation.answer_metrics import export_per_example_csv

        records = [_make_record("e1", "M1", final_answer="Berlin")]
        lookup  = {"e1": _DuckQAGold("Paris", [])}
        path    = export_per_example_csv(records, lookup, str(tmp_path / "out.csv"))
        import csv
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert float(rows[0]["em"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# mean_recall_at_k — QAGold object gold_lookup
# ---------------------------------------------------------------------------

class TestMeanRecallAtKWithObjects:

    def _make_evidence(self, title, sentence_id):
        return {"doc_title": title, "sentence_id": sentence_id,
                "text": "t", "score": 1.0, "rank": 1, "retrieval_query": "q"}

    def test_recall_object_equals_dict_gold(self):
        from src.evaluation.retrieval_metrics import mean_recall_at_k

        ev = [self._make_evidence("France", 0)]
        gold_facts = [{"title": "France", "sentence_id": 0}]
        records = [_make_record("e1", "M4", evidence=ev)]

        dict_lookup = {"e1": {"gold_supporting_facts": gold_facts}}
        obj_lookup  = {"e1": _DuckQAGold("Paris", gold_facts)}

        r_dict = mean_recall_at_k(records, dict_lookup, k=5, method_id="M4")
        r_obj  = mean_recall_at_k(records, obj_lookup, k=5, method_id="M4")

        assert r_obj["mean_recall"] == pytest.approx(r_dict["mean_recall"])

    def test_recall_zero_with_wrong_evidence(self):
        from src.evaluation.retrieval_metrics import mean_recall_at_k

        ev      = [self._make_evidence("Germany", 0)]
        records = [_make_record("e1", "M4", evidence=ev)]
        lookup  = {"e1": _DuckQAGold("Paris", [{"title": "France", "sentence_id": 0}])}

        result = mean_recall_at_k(records, lookup, k=5, method_id="M4")
        assert result["mean_recall"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compare_methods — QAGold object gold_lookup
# ---------------------------------------------------------------------------

class TestCompareMethodsWithObjects:

    def test_compare_with_object_gold(self):
        from src.evaluation.statistical import compare_methods

        records = [
            _make_record("e1", "M1", final_answer="Paris"),
            _make_record("e1", "M2", final_answer="Berlin"),
        ]
        lookup = {"e1": _DuckQAGold("Paris", [])}

        result = compare_methods(records, lookup, "M1", "M2", metric="em",
                                 n_resamples=100, seed=0)
        assert result["mean_a"] == pytest.approx(1.0)
        assert result["mean_b"] == pytest.approx(0.0)

    def test_compare_same_result_dict_vs_object(self):
        from src.evaluation.statistical import compare_methods

        records = [
            _make_record("e1", "M1", final_answer="Paris"),
            _make_record("e1", "M2", final_answer="Paris"),
        ]
        dict_lookup = {"e1": {"gold_answer": "Paris"}}
        obj_lookup  = {"e1": _DuckQAGold("Paris", [])}

        r_dict = compare_methods(records, dict_lookup, "M1", "M2", n_resamples=100, seed=0)
        r_obj  = compare_methods(records, obj_lookup,  "M1", "M2", n_resamples=100, seed=0)

        assert r_dict["delta_obs"] == pytest.approx(r_obj["delta_obs"])


# ---------------------------------------------------------------------------
# contracts.schemas.QAGold (skipped when contracts not installed)
# ---------------------------------------------------------------------------

class TestContractsQAGold:

    def test_em_f1_with_contracts_qaGold(self):
        contracts = pytest.importorskip("contracts.schemas")
        from src.evaluation.answer_metrics import compute_em_f1_batch

        gold = contracts.QAGold(
            example_id="e1",
            gold_answer="Paris",
            gold_supporting_facts=[{"title": "France", "sentence_id": 0}],
        )
        records = [_make_record("e1", "M1", final_answer="Paris")]
        result  = compute_em_f1_batch(records, {"e1": gold})
        assert result["M1"]["em"] == pytest.approx(1.0)

    def test_recall_with_contracts_qaGold(self):
        contracts = pytest.importorskip("contracts.schemas")
        from src.evaluation.retrieval_metrics import mean_recall_at_k

        ev   = [{"doc_title": "France", "sentence_id": 0,
                 "text": "t", "score": 1.0, "rank": 1, "retrieval_query": "q"}]
        gold = contracts.QAGold(
            example_id="e1",
            gold_answer="Paris",
            gold_supporting_facts=[{"title": "France", "sentence_id": 0}],
        )
        records = [_make_record("e1", "M4", evidence=ev)]
        result  = mean_recall_at_k(records, {"e1": gold}, k=5, method_id="M4")
        assert result["mean_recall"] == pytest.approx(1.0)
