"""Tests for to_contract_record() in src/runner/records.py.

Requires contracts.schemas (from origin/feat/contracts-v0.1) to be on sys.path.
All tests are skipped when contracts is not importable.  They become active on
the integration branch where all sub-branches are merged.
"""

import pytest


def _make_m1_record():
    from src.runner.records import ExperimentRecord
    return ExperimentRecord(
        example_id="e1",
        method_id="M1",
        status="ok",
        timestamp="2026-10-29T00:00:00+00:00",
        draft={"draft_id": "d-abc", "short_answer": "Paris", "explanation": "Capital"},
        final=None,
        evidence={},
        usage=[
            {"stage": "generate_draft", "model": "gpt-4o-mini",
             "input_tokens": 100, "output_tokens": 50,
             "cost_usd": 0.001, "latency_seconds": 0.5, "cache_hit": False}
        ],
        model_config={"model": "gpt-4o-mini"},
    )


def _make_m4_record():
    from src.runner.records import ExperimentRecord
    return ExperimentRecord(
        example_id="e2",
        method_id="M4",
        status="ok",
        timestamp="2026-10-29T00:00:00+00:00",
        draft={"draft_id": "d-xyz", "short_answer": "Berlin", "explanation": "Capital"},
        final={"draft_id": "d-xyz", "feedback_id": "f-1",
               "final_answer": "Berlin", "final_explanation": "Capital city"},
        claims=[],
        evidence={"d-xyz": [
            {"doc_title": "Berlin", "sentence_id": 0, "text": "Berlin is ...",
             "score": 0.9, "retrieval_query": "q", "rank": 1}
        ]},
        feedback=None,
        usage=[],
        model_config={"model": "gpt-4o-mini"},
    )


class TestToContractRecord:
    def test_function_is_importable(self):
        """to_contract_record must be exported from records module."""
        from src.runner.records import to_contract_record  # noqa: F401

    def test_m1_final_is_none(self):
        """M1 conversion: final must be None (no RevisionResult)."""
        contracts = pytest.importorskip("contracts.schemas")
        from src.runner.records import to_contract_record
        rec = _make_m1_record()
        result = to_contract_record(rec)
        assert result.final is None

    def test_m1_draft_converted(self):
        """M1 draft dict is converted to AnswerDraft object."""
        contracts = pytest.importorskip("contracts.schemas")
        from src.runner.records import to_contract_record
        rec = _make_m1_record()
        result = to_contract_record(rec)
        assert isinstance(result.draft, contracts.AnswerDraft)
        assert result.draft.short_answer == "Paris"
        assert result.draft.draft_id == "d-abc"

    def test_m4_final_converted(self):
        """M4 final dict is converted to RevisionResult object."""
        contracts = pytest.importorskip("contracts.schemas")
        from src.runner.records import to_contract_record
        rec = _make_m4_record()
        result = to_contract_record(rec)
        assert isinstance(result.final, contracts.RevisionResult)
        assert result.final.final_answer == "Berlin"

    def test_m4_evidence_converted(self):
        """Evidence dict is preserved with Evidence objects as values."""
        contracts = pytest.importorskip("contracts.schemas")
        from src.runner.records import to_contract_record
        rec = _make_m4_record()
        result = to_contract_record(rec)
        assert "d-xyz" in result.evidence
        ev_list = result.evidence["d-xyz"]
        assert len(ev_list) == 1
        assert isinstance(ev_list[0], contracts.Evidence)
        assert ev_list[0].doc_title == "Berlin"

    def test_usage_converted_to_llm_usage(self):
        """Usage dicts are converted to LLMUsage objects."""
        contracts = pytest.importorskip("contracts.schemas")
        from src.runner.records import to_contract_record
        rec = _make_m1_record()
        result = to_contract_record(rec)
        assert len(result.usage) == 1
        assert isinstance(result.usage[0], contracts.LLMUsage)
        assert result.usage[0].stage == "generate_draft"
        assert result.usage[0].cache_hit is False

    def test_model_config_preserved(self):
        """model_config dict is passed through unchanged."""
        contracts = pytest.importorskip("contracts.schemas")
        from src.runner.records import to_contract_record
        rec = _make_m1_record()
        result = to_contract_record(rec)
        assert result.model_config == {"model": "gpt-4o-mini"}

    def test_example_id_and_method_id_preserved(self):
        """Identity fields are copied verbatim."""
        contracts = pytest.importorskip("contracts.schemas")
        from src.runner.records import to_contract_record
        rec = _make_m1_record()
        result = to_contract_record(rec)
        assert result.example_id == "e1"
        assert result.method_id == "M1"
        assert result.status == "ok"

    def test_error_record_converted(self):
        """Error records convert with final=None, draft=None."""
        contracts = pytest.importorskip("contracts.schemas")
        from src.runner.records import ExperimentRecord, to_contract_record
        rec = ExperimentRecord(
            example_id="e3", method_id="M2", status="error",
            timestamp="2026-10-29T00:00:00+00:00",
            error="LLM timeout",
        )
        result = to_contract_record(rec)
        assert result.status == "error"
        assert result.draft is None
        assert result.final is None
        assert result.error == "LLM timeout"
