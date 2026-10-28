"""Unit tests for src/evaluation/retrieval_metrics.py.

Gold facts use key 'title'; retrieved evidence uses key 'doc_title'.
This asymmetry mirrors B's actual data format and is tested explicitly.
No external data or API calls are used.
"""

import pytest
from src.evaluation.retrieval_metrics import (
    recall_at_k,
    full_coverage_at_k,
    mean_recall_at_k,
    RecallDenominator,
)


def make_gold(facts: list[tuple[str, int]]) -> list[dict]:
    """Build gold_supporting_facts list using B's 'title' key."""
    return [{"title": t, "sentence_id": s} for t, s in facts]


def make_evidence(items: list[tuple[str, int]], k_total: int = None) -> list[dict]:
    """Build retrieved evidence list using B's 'doc_title' key."""
    result = []
    for rank, (doc_title, sentence_id) in enumerate(items, start=1):
        result.append({
            "doc_title": doc_title,
            "sentence_id": sentence_id,
            "text": f"Text for {doc_title}[{sentence_id}]",
            "score": 1.0 - 0.1 * rank,
            "retrieval_query": "test query",
            "rank": rank,
        })
    return result


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:
    def test_full_recall_k5(self):
        gold = make_gold([("France", 0), ("Germany", 1)])
        evidence = make_evidence([("France", 0), ("Germany", 1), ("Italy", 0)])
        assert recall_at_k(evidence, gold, k=5) == pytest.approx(1.0)

    def test_partial_recall(self):
        gold = make_gold([("France", 0), ("Germany", 1)])
        evidence = make_evidence([("France", 0), ("Italy", 0)])
        # 1 of 2 gold facts found
        assert recall_at_k(evidence, gold, k=5) == pytest.approx(0.5)

    def test_zero_recall(self):
        gold = make_gold([("France", 0)])
        evidence = make_evidence([("Germany", 1)])
        assert recall_at_k(evidence, gold, k=5) == pytest.approx(0.0)

    def test_cutoff_k_respected(self):
        gold = make_gold([("France", 0)])
        # gold fact is at rank 3, k=2 should miss it
        evidence = make_evidence([("Germany", 0), ("Italy", 0), ("France", 0)])
        assert recall_at_k(evidence, gold, k=2) == pytest.approx(0.0)
        assert recall_at_k(evidence, gold, k=3) == pytest.approx(1.0)

    def test_empty_gold_returns_zero(self):
        evidence = make_evidence([("France", 0)])
        assert recall_at_k(evidence, [], k=5) == pytest.approx(0.0)

    def test_empty_evidence_returns_zero(self):
        gold = make_gold([("France", 0)])
        assert recall_at_k([], gold, k=5) == pytest.approx(0.0)

    def test_title_doc_title_asymmetry_explicit(self):
        """Gold uses 'title'; evidence uses 'doc_title'. Must match on value."""
        gold = [{"title": "France", "sentence_id": 0}]
        evidence = [{"doc_title": "France", "sentence_id": 0,
                     "text": "t", "score": 0.9, "retrieval_query": "q", "rank": 1}]
        assert recall_at_k(evidence, gold, k=1) == pytest.approx(1.0)

    def test_question_conditioned_mode(self):
        gold = make_gold([("France", 0)])
        evidence = make_evidence([("France", 0)])
        r = recall_at_k(
            evidence, gold, k=5,
            denominator=RecallDenominator.QUESTION_CONDITIONED
        )
        assert r == pytest.approx(1.0)

    def test_claim_conditioned_mode(self):
        # Claim-conditioned uses the same formula; caller supplies the
        # subset of gold facts relevant to the claim.
        claim_gold = make_gold([("France", 0)])
        evidence = make_evidence([("France", 0)])
        r = recall_at_k(
            evidence, claim_gold, k=5,
            denominator=RecallDenominator.CLAIM_CONDITIONED
        )
        assert r == pytest.approx(1.0)

    def test_duplicate_evidence_counted_once(self):
        gold = make_gold([("France", 0)])
        # Same sentence appears twice in evidence; set deduplication applies
        evidence = make_evidence([("France", 0), ("France", 0)])
        assert recall_at_k(evidence, gold, k=5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# full_coverage_at_k
# ---------------------------------------------------------------------------

class TestFullCoverageAtK:
    def test_full_coverage(self):
        gold = make_gold([("France", 0), ("Germany", 1)])
        evidence = make_evidence([("France", 0), ("Germany", 1)])
        assert full_coverage_at_k(evidence, gold, k=5) is True

    def test_partial_not_full(self):
        gold = make_gold([("France", 0), ("Germany", 1)])
        evidence = make_evidence([("France", 0)])
        assert full_coverage_at_k(evidence, gold, k=5) is False

    def test_empty_gold_returns_false(self):
        evidence = make_evidence([("France", 0)])
        assert full_coverage_at_k(evidence, [], k=5) is False


# ---------------------------------------------------------------------------
# mean_recall_at_k
# ---------------------------------------------------------------------------

class TestMeanRecallAtK:
    def _records(self, rows):
        return [
            {
                "example_id": r["eid"],
                "method_id": r["mid"],
                "status": r.get("status", "ok"),
                "evidence": r.get("evidence", []),
            }
            for r in rows
        ]

    def _gold(self, mapping):
        return {
            eid: {"gold_supporting_facts": make_gold(facts)}
            for eid, facts in mapping.items()
        }

    def test_perfect_recall(self):
        evidence = make_evidence([("France", 0)])
        records = self._records([
            {"eid": "e1", "mid": "M4", "evidence": evidence},
            {"eid": "e2", "mid": "M4", "evidence": evidence},
        ])
        gold = self._gold({"e1": [("France", 0)], "e2": [("France", 0)]})
        result = mean_recall_at_k(records, gold, k=5, method_id="M4")
        assert result["mean_recall"] == pytest.approx(1.0)
        assert result["n"] == 2
        assert result["n_ok"] == 2

    def test_error_record_counts_in_denominator(self):
        evidence = make_evidence([("France", 0)])
        records = self._records([
            {"eid": "e1", "mid": "M4", "evidence": evidence},
            {"eid": "e2", "mid": "M4", "status": "error"},
        ])
        gold = self._gold({"e1": [("France", 0)], "e2": [("France", 0)]})
        result = mean_recall_at_k(records, gold, k=5, method_id="M4")
        assert result["n"] == 2
        assert result["n_error"] == 1
        # recall_sum=1.0, denominator=2
        assert result["mean_recall"] == pytest.approx(0.5)

    def test_filters_by_method_id(self):
        evidence = make_evidence([("France", 0)])
        records = self._records([
            {"eid": "e1", "mid": "M4", "evidence": evidence},
            {"eid": "e1", "mid": "M5", "evidence": []},
        ])
        gold = self._gold({"e1": [("France", 0)]})
        r_m4 = mean_recall_at_k(records, gold, k=5, method_id="M4")
        r_m5 = mean_recall_at_k(records, gold, k=5, method_id="M5")
        assert r_m4["mean_recall"] == pytest.approx(1.0)
        assert r_m5["mean_recall"] == pytest.approx(0.0)

    def test_denominator_label_preserved(self):
        records = self._records([{"eid": "e1", "mid": "M4", "evidence": []}])
        gold = self._gold({"e1": []})
        # QUESTION_CONDITIONED is valid; CLAIM_CONDITIONED now raises (see below)
        result = mean_recall_at_k(
            records, gold, k=5, method_id="M4",
            denominator=RecallDenominator.QUESTION_CONDITIONED
        )
        assert result["denominator"] == "question_conditioned"

    def test_empty_records(self):
        result = mean_recall_at_k([], {}, k=5, method_id="M4")
        assert result["mean_recall"] == pytest.approx(0.0)
        assert result["n"] == 0


# ---------------------------------------------------------------------------
# CLAIM_CONDITIONED raises NotImplementedError (Issue 6)
# ---------------------------------------------------------------------------

class TestClaimConditionedNotImplemented:
    def test_mean_recall_raises_for_claim_conditioned(self):
        records = [{"example_id": "e1", "method_id": "M4",
                    "status": "ok", "evidence": []}]
        gold = {"e1": {"gold_supporting_facts": []}}
        with pytest.raises(NotImplementedError, match="claim.*gold|GAP-7"):
            mean_recall_at_k(
                records, gold, k=5, method_id="M4",
                denominator=RecallDenominator.CLAIM_CONDITIONED
            )


# ---------------------------------------------------------------------------
# Tuple gold_supporting_facts (contracts.QAGold format)
# ---------------------------------------------------------------------------

class TestTupleGoldSupportingFacts:
    def test_tuple_format_accepted(self):
        """contracts.QAGold uses list[tuple[str, int]] for gold_supporting_facts."""
        gold_tuples = [("France", 0), ("Germany", 1)]
        evidence = make_evidence([("France", 0), ("Germany", 1)])
        result = recall_at_k(evidence, gold_tuples, k=5)
        assert result == pytest.approx(1.0)

    def test_tuple_partial_recall(self):
        gold_tuples = [("France", 0), ("Germany", 1)]
        evidence = make_evidence([("France", 0)])
        result = recall_at_k(evidence, gold_tuples, k=5)
        assert result == pytest.approx(0.5)

    def test_mixed_dict_and_tuple_format(self):
        """Mixed list: some dicts, some tuples."""
        gold_mixed = [{"title": "France", "sentence_id": 0}, ("Germany", 1)]
        evidence = make_evidence([("France", 0), ("Germany", 1)])
        result = recall_at_k(evidence, gold_mixed, k=5)
        assert result == pytest.approx(1.0)

    def test_tuple_full_coverage(self):
        gold_tuples = [("France", 0)]
        evidence = make_evidence([("France", 0)])
        assert full_coverage_at_k(evidence, gold_tuples, k=5) is True

    def test_tuple_zero_recall(self):
        gold_tuples = [("France", 0)]
        evidence = make_evidence([("Germany", 1)])
        assert recall_at_k(evidence, gold_tuples, k=5) == pytest.approx(0.0)
