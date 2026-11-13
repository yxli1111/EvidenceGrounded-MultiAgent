"""Offline safety tests for claim distribution real mode.

All LLM interactions are mocked.
No real OpenRouter API requests are made.
"""

import json
from unittest.mock import MagicMock

import pytest

from contracts.schemas import AnswerDraft
from src.claims.extractor import ClaimExtractionError
from src.core.budget import BudgetExceededError
from src.core.llm_client import LLMClientError

from scripts.analyze_claim_distribution import (
    run_batch,
    prepare_run,
    make_fingerprint,
    build_parser,
)


def make_draft(draft_id, explanation="A factual statement."):
    return AnswerDraft(
        draft_id=draft_id,
        short_answer="Test",
        explanation=explanation,
    )


def make_claims(count):
    """Create mock claim objects."""
    return [MagicMock() for _ in range(count)]


def test_budget_exhaustion_stops_batch(tmp_path):
    """Budget exhaustion must stop subsequent drafts."""

    drafts = [
        make_draft("d1"),
        make_draft("d2"),
        make_draft("d3"),
    ]

    extractor = MagicMock()

    extractor.extract_claims.side_effect = [
        make_claims(2),
        BudgetExceededError("Budget exhausted"),
    ]

    llm = MagicMock()
    output = tmp_path / "counts.csv"

    records = run_batch(
        drafts=drafts,
        extractor=extractor,
        llm=llm,
        output=output,
        max_examples=3,
    )

    assert len(records) == 1
    assert records[0]["draft_id"] == "d1"
    assert records[0]["claim_count"] == 2

    # d3 must not be processed.
    assert extractor.extract_claims.call_count == 2
    assert output.exists()


def test_api_failure_is_not_zero_claims(tmp_path):
    """An API failure must not become a valid zero count."""

    extractor = MagicMock()
    extractor.extract_claims.side_effect = (
        LLMClientError("Mock API failure")
    )

    output = tmp_path / "counts.csv"

    records = run_batch(
        drafts=[make_draft("d1")],
        extractor=extractor,
        llm=MagicMock(),
        output=output,
        max_examples=1,
    )

    assert len(records) == 1
    assert records[0]["status"] == "error"
    assert records[0]["claim_count"] is None
    assert records[0]["error_type"] == "LLMClientError"


def test_claim_extraction_failure_recorded(tmp_path):
    """Malformed extraction results must be recorded as errors."""

    extractor = MagicMock()
    extractor.extract_claims.side_effect = (
        ClaimExtractionError("Invalid JSON")
    )

    records = run_batch(
        drafts=[make_draft("d1")],
        extractor=extractor,
        llm=MagicMock(),
        output=tmp_path / "counts.csv",
        max_examples=1,
    )

    assert records[0]["status"] == "error"
    assert records[0]["claim_count"] is None
    assert records[0]["error_type"] == "ClaimExtractionError"


def test_resume_skips_successful_drafts(tmp_path):
    """Previously completed drafts must not trigger new calls."""

    extractor = MagicMock()
    extractor.extract_claims.return_value = make_claims(3)

    existing = [{
        "draft_id": "d1",
        "status": "success",
        "claim_count": 2,
        "error_type": "",
    }]

    records = run_batch(
        drafts=[
            make_draft("d1"),
            make_draft("d2"),
        ],
        extractor=extractor,
        llm=MagicMock(),
        output=tmp_path / "counts.csv",
        max_examples=2,
        existing=existing,
    )

    assert len(records) == 2
    assert records[0]["claim_count"] == 2
    assert records[1]["claim_count"] == 3

    # Only d2 should be processed.
    extractor.extract_claims.assert_called_once()
    assert (
        extractor.extract_claims.call_args.args[0].draft_id
        == "d2"
    )


def test_resume_skips_failed_drafts(tmp_path):
    """Failed drafts are not retried automatically."""

    extractor = MagicMock()

    existing = [{
        "draft_id": "d1",
        "status": "error",
        "claim_count": None,
        "error_type": "LLMClientError",
    }]

    records = run_batch(
        drafts=[make_draft("d1")],
        extractor=extractor,
        llm=MagicMock(),
        output=tmp_path / "counts.csv",
        max_examples=1,
        existing=existing,
    )

    assert len(records) == 1
    extractor.extract_claims.assert_not_called()


def test_resume_rejects_changed_configuration(tmp_path):
    """Resume must reject a different experiment fingerprint."""

    output = tmp_path / "counts.csv"
    meta = tmp_path / "counts.csv.meta.json"

    output.write_text(
        "draft_id,status,claim_count,error_type\n"
        "d1,success,2,\n",
        encoding="utf-8",
    )

    meta.write_text(
        json.dumps({
            "fingerprint": "original-fingerprint",
            "format_version": 1,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="changed"):
        prepare_run(
            output,
            fingerprint="different-fingerprint",
            resume=True,
        )


def test_real_mode_requires_confirmation():
    """The CLI must not silently enable real API mode."""

    parser = build_parser()

    args = parser.parse_args([
        "--mode", "real",
        "--input", "drafts.json",
        "--output", "counts.csv",
    ])

    assert args.mode == "real"
    assert args.confirm_real is False


def test_mock_mode_is_default():
    """Mock must remain the default mode."""

    parser = build_parser()

    args = parser.parse_args([
        "--input", "drafts.json",
        "--output", "counts.csv",
    ])

    assert args.mode == "mock"
    assert args.confirm_real is False


def test_successful_empty_claims_is_valid(tmp_path):
    """A genuine successful empty result must be zero."""

    extractor = MagicMock()
    extractor.extract_claims.return_value = []

    records = run_batch(
        drafts=[make_draft("d1")],
        extractor=extractor,
        llm=MagicMock(),
        output=tmp_path / "counts.csv",
        max_examples=1,
    )

    assert records[0]["status"] == "success"
    assert records[0]["claim_count"] == 0

def test_real_mode_without_confirmation_never_initializes_client(
    monkeypatch,
    tmp_path,
):
    """Real mode must stop before initializing the API client."""
    import sys
    from scripts import analyze_claim_distribution as module

    def forbidden_client(*args, **kwargs):
        raise AssertionError(
            "OpenRouter client must not be initialized"
        )

    monkeypatch.setattr(
        module,
        "OpenRouterLLMClient",
        forbidden_client,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_claim_distribution.py",
            "--mode", "real",
            "--input", str(tmp_path / "drafts.json"),
            "--output", str(tmp_path / "counts.csv"),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        module.main()

    assert exc_info.value.code == 2
    assert not (tmp_path / "counts.csv").exists()


def test_resume_rejects_missing_csv(tmp_path):
    """Resume must reject missing CSV even if metadata exists."""
    from scripts.analyze_claim_distribution import prepare_run

    output = tmp_path / "counts.csv"
    meta = tmp_path / "counts.csv.meta.json"

    meta.write_text(
        json.dumps({
            "fingerprint": "test-fingerprint",
            "format_version": 1,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires CSV"):
        prepare_run(output, "test-fingerprint", resume=True)


def test_resume_rejects_duplicate_draft_ids(tmp_path):
    """Duplicate rows must not be silently accepted."""
    from scripts.analyze_claim_distribution import load_existing_records

    output = tmp_path / "counts.csv"
    output.write_text(
        "draft_id,status,claim_count,error_type\n"
        "d1,success,2,\n"
        "d1,success,3,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate draft_id"):
        load_existing_records(output)


def test_resume_rejects_invalid_claim_count(tmp_path):
    """Invalid successful claim counts must be rejected."""
    from scripts.analyze_claim_distribution import load_existing_records

    output = tmp_path / "counts.csv"
    output.write_text(
        "draft_id,status,claim_count,error_type\n"
        "d1,success,-1,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Negative claim count"):
        load_existing_records(output)


def test_resume_rejects_unknown_status(tmp_path):
    """Unknown statuses must not be treated as success."""
    from scripts.analyze_claim_distribution import load_existing_records

    output = tmp_path / "counts.csv"
    output.write_text(
        "draft_id,status,claim_count,error_type\n"
        "d1,pending,2,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid status"):
        load_existing_records(output)