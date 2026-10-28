"""Tests for configurable retrieval parameters in ExperimentRunner.

Covers:
- m4_top_k / m5_top_k_per_claim separate from global top_k
- query_include_question template (False = text only, True = question + \\n + text)
- m5_total_evidence_budget round-robin allocation across claims
- retrieval params written into model_settings on every record

No real API calls; uses mock ports from src.runner.mocks.
"""

import pytest

from src.runner.mocks import (
    MockClaimsPort,
    MockDraftPort,
    MockLLMClient,
    MockRetrievePort,
    MockVerificationPort,
    make_synthetic_example,
)
from src.runner.records import load_records
from src.runner.runner import ExperimentRunner


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_runner(tmp_path, **kwargs):
    """Return (runner, output_file) using mock ports with kwargs passed to runner."""
    ports = dict(
        draft_port=MockDraftPort(),
        retrieve_port=MockRetrievePort(),
        claims_port=MockClaimsPort(),
        verification_port=MockVerificationPort(),
        llm=MockLLMClient(),
        top_k=3,
    )
    ports.update(kwargs)
    runner = ExperimentRunner(**ports)
    out = str(tmp_path / "out.jsonl")
    return runner, out


# ---------------------------------------------------------------------------
# Separate top-k per method
# ---------------------------------------------------------------------------

class TestSeparateTopK:
    def test_m4_top_k_independent_of_global(self, tmp_path):
        """m4_top_k overrides global top_k for M4 retrieval."""
        received = []

        class RecordingRetrievePort(MockRetrievePort):
            def retrieve(self, query, context, top_k):
                received.append(top_k)
                return super().retrieve(query, context, top_k)

        runner, out = _make_runner(
            tmp_path,
            retrieve_port=RecordingRetrievePort(),
            top_k=3,
            m4_top_k=7,
        )
        runner.run([make_synthetic_example("e1")], out)
        # M4 does one retrieve call; M5 does retrieve per claim
        m4_call = received[0]
        assert m4_call == 7, f"M4 should use m4_top_k=7, got {m4_call}"

    def test_m5_top_k_per_claim_independent_of_global(self, tmp_path):
        """m5_top_k_per_claim overrides global top_k for each M5 claim retrieve."""
        received = []

        class RecordingRetrievePort(MockRetrievePort):
            def retrieve(self, query, context, top_k):
                received.append(top_k)
                return super().retrieve(query, context, top_k)

        runner, out = _make_runner(
            tmp_path,
            retrieve_port=RecordingRetrievePort(),
            top_k=3,
            m5_top_k_per_claim=6,
        )
        runner.run([make_synthetic_example("e1")], out)
        records = {r.method_id: r for r in load_records(out)}
        n_claims = len(records["M5"].claims)
        # M4 = received[0], M5 = received[1..n_claims]
        m5_calls = received[1:1 + n_claims]
        assert all(k == 6 for k in m5_calls), (
            f"M5 should use m5_top_k_per_claim=6, got {m5_calls}"
        )

    def test_global_top_k_used_when_no_separate_m4(self, tmp_path):
        received = []

        class RecordingRetrievePort(MockRetrievePort):
            def retrieve(self, query, context, top_k):
                received.append(top_k)
                return super().retrieve(query, context, top_k)

        runner, out = _make_runner(
            tmp_path,
            retrieve_port=RecordingRetrievePort(),
            top_k=4,
            # no m4_top_k set
        )
        runner.run([make_synthetic_example("e1")], out)
        assert received[0] == 4


# ---------------------------------------------------------------------------
# query_include_question template
# ---------------------------------------------------------------------------

class TestQueryTemplate:
    def _get_queries(self, tmp_path, query_include_question, example=None):
        received = []

        class RecordingRetrievePort(MockRetrievePort):
            def retrieve(self, query, context, top_k):
                received.append(query)
                return super().retrieve(query, context, top_k)

        if example is None:
            example = make_synthetic_example("e1", question="What is France?")
        runner, out = _make_runner(
            tmp_path,
            retrieve_port=RecordingRetrievePort(),
            query_include_question=query_include_question,
        )
        runner.run([example], out)
        return received

    def test_default_query_is_explanation_only_m4(self, tmp_path):
        """Default: M4 query = explanation (no question prefix)."""
        queries = self._get_queries(tmp_path, query_include_question=False)
        m4_query = queries[0]
        assert "What is France?" not in m4_query
        assert "mock explanation" in m4_query.lower()

    def test_query_include_question_true_m4_prefix(self, tmp_path):
        """query_include_question=True: M4 query = question + '\\n' + explanation."""
        queries = self._get_queries(tmp_path, query_include_question=True)
        m4_query = queries[0]
        assert m4_query.startswith("What is France?\n"), (
            f"M4 query should start with question+newline, got: {m4_query!r}"
        )

    def test_query_include_question_true_m5_prefix(self, tmp_path):
        """query_include_question=True: M5 query = question + '\\n' + claim_text."""
        queries = self._get_queries(tmp_path, query_include_question=True)
        # queries[0] = M4, queries[1..] = M5 per claim
        for m5_query in queries[1:]:
            assert m5_query.startswith("What is France?\n"), (
                f"M5 query should start with question+newline, got: {m5_query!r}"
            )

    def test_m4_and_m5_use_same_template(self, tmp_path):
        """M4 and M5 must use the same template (question + '\\n' + text)."""
        queries = self._get_queries(tmp_path, query_include_question=True)
        # All queries should start with question+newline
        for q in queries:
            assert "\n" in q, f"Expected newline in query: {q!r}"


# ---------------------------------------------------------------------------
# m5_total_evidence_budget — round-robin allocation
# ---------------------------------------------------------------------------

class TestM5EvidenceBudget:
    def test_budget_limits_total_top_k_calls(self, tmp_path):
        """With budget, total evidence requested ≤ budget (per round-robin)."""
        allocations = []

        class RecordingRetrievePort(MockRetrievePort):
            def retrieve(self, query, context, top_k):
                allocations.append(top_k)
                return super().retrieve(query, context, top_k)

        runner, out = _make_runner(
            tmp_path,
            retrieve_port=RecordingRetrievePort(),
            top_k=5,
            m5_top_k_per_claim=5,
            m5_total_evidence_budget=3,
        )
        runner.run([make_synthetic_example("e1")], out)
        records = {r.method_id: r for r in load_records(out)}
        n_claims = len(records["M5"].claims)
        m5_allocs = allocations[1:1 + n_claims]
        assert sum(m5_allocs) <= 3, (
            f"Total evidence with budget=3 should be ≤ 3, got {sum(m5_allocs)}"
        )

    def test_no_budget_uses_full_top_k_per_claim(self, tmp_path):
        """Without budget, each claim gets full m5_top_k_per_claim."""
        allocations = []

        class RecordingRetrievePort(MockRetrievePort):
            def retrieve(self, query, context, top_k):
                allocations.append(top_k)
                return super().retrieve(query, context, top_k)

        runner, out = _make_runner(
            tmp_path,
            retrieve_port=RecordingRetrievePort(),
            top_k=5,
            m5_top_k_per_claim=4,
            m5_total_evidence_budget=None,
        )
        runner.run([make_synthetic_example("e1")], out)
        records = {r.method_id: r for r in load_records(out)}
        n_claims = len(records["M5"].claims)
        m5_allocs = allocations[1:1 + n_claims]
        assert all(k == 4 for k in m5_allocs), (
            f"Without budget, all claim retrievals should use top_k=4, got {m5_allocs}"
        )

    def test_evidence_by_claim_preserved_under_budget(self, tmp_path):
        """evidence_by_claim must contain every claim_id even with tight budget."""
        runner, out = _make_runner(
            tmp_path,
            m5_total_evidence_budget=1,  # very tight: only 1 slot total
        )
        runner.run([make_synthetic_example("e1")], out)
        records = {r.method_id: r for r in load_records(out)}
        m5 = records["M5"]
        claim_ids = {c["claim_id"] for c in m5.claims}
        assert set(m5.evidence_by_claim.keys()) == claim_ids, (
            "evidence_by_claim must contain all claim_ids even with tight budget"
        )


# ---------------------------------------------------------------------------
# Retrieval params in model_settings
# ---------------------------------------------------------------------------

class TestModelSettingsRetrieval:
    def test_retrieval_params_written_to_model_settings(self, tmp_path):
        """Retrieval params must appear in every record's model_settings."""
        runner, out = _make_runner(
            tmp_path,
            top_k=3,
            m4_top_k=5,
            m5_top_k_per_claim=2,
            m5_total_evidence_budget=4,
            query_include_question=True,
        )
        runner.run(
            [make_synthetic_example("e1")],
            out,
            model_settings={"model": "test-model"},
        )
        records = load_records(out)
        for rec in records:
            ms = rec.model_settings
            assert ms is not None
            ret = ms.get("_retrieval")
            assert ret is not None, f"model_settings missing '_retrieval' key for {rec.method_id}"
            assert ret["m4_top_k"] == 5
            assert ret["m5_top_k_per_claim"] == 2
            assert ret["m5_total_evidence_budget"] == 4
            assert ret["query_include_question"] is True

    def test_model_settings_preserved_when_no_retrieval_params(self, tmp_path):
        """Caller's model_settings must be preserved; _retrieval key added."""
        runner, out = _make_runner(tmp_path)
        runner.run(
            [make_synthetic_example("e1")],
            out,
            model_settings={"model": "my-model", "temperature": 0.0},
        )
        records = load_records(out)
        for rec in records:
            assert rec.model_settings["model"] == "my-model"
            assert "_retrieval" in rec.model_settings


# ---------------------------------------------------------------------------
# exclude_ids safety check
# ---------------------------------------------------------------------------

class TestExcludeIds:
    def test_exclude_ids_raises_on_overlap(self, tmp_path):
        """run() must raise if any example_id appears in exclude_ids."""
        runner, out = _make_runner(tmp_path)
        example = make_synthetic_example("forbidden_id")
        with pytest.raises(ValueError, match="forbidden_id"):
            runner.run([example], out, exclude_ids={"forbidden_id"})

    def test_exclude_ids_no_overlap_runs_normally(self, tmp_path):
        runner, out = _make_runner(tmp_path)
        example = make_synthetic_example("safe_id")
        records = runner.run([example], out, exclude_ids={"other_id"})
        assert len(records) == 5

    def test_exclude_ids_empty_set_runs_normally(self, tmp_path):
        runner, out = _make_runner(tmp_path)
        example = make_synthetic_example("e1")
        records = runner.run([example], out, exclude_ids=set())
        assert len(records) == 5

    def test_exclude_ids_none_runs_normally(self, tmp_path):
        runner, out = _make_runner(tmp_path)
        example = make_synthetic_example("e1")
        records = runner.run([example], out, exclude_ids=None)
        assert len(records) == 5

    def test_exclude_ids_lists_all_overlapping_ids_in_error(self, tmp_path):
        runner, out = _make_runner(tmp_path)
        examples = [
            make_synthetic_example("bad1"),
            make_synthetic_example("bad2"),
            make_synthetic_example("good"),
        ]
        with pytest.raises(ValueError) as exc_info:
            runner.run(examples, out, exclude_ids={"bad1", "bad2"})
        msg = str(exc_info.value)
        assert "bad1" in msg and "bad2" in msg
