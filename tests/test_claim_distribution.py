from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from contracts.schemas import AnswerDraft
from src.claims.extractor import AtomicClaimExtractor
from src.core.llm_client import LLMClientError

from scripts.analyze_claim_distribution import (
    analyze,
    summarize,
    percentile_nearest_rank,
    save_csv,
    load_drafts,
)


def make_draft(draft_id, explanation):
    return AnswerDraft(
        draft_id=draft_id,
        short_answer="Test",
        explanation=explanation,
    )


def test_distribution_statistics():
    records = [
        {
            "draft_id": str(i),
            "status": "success",
            "claim_count": count,
            "error_type": "",
        }
        for i, count in enumerate([1, 2, 2, 3, 5])
    ]

    result = summarize(records)

    assert result["mean"] == pytest.approx(2.6)
    assert result["median"] == 2
    assert result["p90"] == 5
    assert result["max"] == 5


def test_failed_samples_excluded():
    records = [
        {
            "draft_id": "a",
            "status": "success",
            "claim_count": 2,
            "error_type": "",
        },
        {
            "draft_id": "b",
            "status": "error",
            "claim_count": None,
            "error_type": "LLMClientError",
        },
    ]

    result = summarize(records)

    assert result["success"] == 1
    assert result["failed"] == 1
    assert result["mean"] == 2


def test_empty_results():
    result = summarize([])

    assert result["mean"] is None
    assert result["p90"] is None


def test_percentile():
    assert percentile_nearest_rank([1, 2, 3, 4, 5], 90) == 5


def test_batch_extraction():
    extractor = AtomicClaimExtractor()

    llm = MagicMock()
    llm.generate.return_value = SimpleNamespace(
        text='{"claims": ["Paris is in France."]}'
    )

    records = analyze(
        [make_draft("a", "Paris is in France.")],
        extractor,
        llm,
        max_examples=1,
    )

    assert records[0]["claim_count"] == 1
    assert records[0]["status"] == "success"


def test_batch_failure_recorded():
    extractor = AtomicClaimExtractor()

    llm = MagicMock()
    llm.generate.side_effect = LLMClientError(
        "Mock failure"
    )

    records = analyze(
        [make_draft("a", "Paris is in France.")],
        extractor,
        llm,
        max_examples=1,
    )

    assert records[0]["status"] == "error"
    assert records[0]["claim_count"] is None


def test_csv_output(tmp_path):
    path = tmp_path / "counts.csv"
    records = [{
        "draft_id": "a",
        "status": "success",
        "claim_count": 2,
        "error_type": "",
    }]

    save_csv(records, path)

    assert path.exists()
    assert "claim_count" in path.read_text(
        encoding="utf-8"
    )


def test_load_drafts(tmp_path):
    path = tmp_path / "drafts.json"
    path.write_text(
        '[{"draft_id":"a","short_answer":"Paris",'
        '"explanation":"Paris is in France."}]',
        encoding="utf-8",
    )

    drafts = load_drafts(path)

    assert len(drafts) == 1
    assert drafts[0].draft_id == "a"


def test_mock_empty_explanation_does_not_match_other_drafts():
    """Empty explanations must not match unrelated drafts."""
    import json
    from scripts.analyze_claim_distribution import MockLLMClient

    # The empty explanation is intentionally placed first.
    llm = MockLLMClient({
        "": [],
        "Apollo 11 landed on the Moon.": [
            "Apollo 11 landed on the Moon."
        ],
    })

    response = llm.generate(
        stage="claim_extraction",
        messages=[
            {
                "role": "user",
                "content": "Apollo 11 landed on the Moon.",
            }
        ],
        model="mock-model",
        temperature=0,
        max_tokens=128,
    )

    data = json.loads(response.text)

    assert len(data["claims"]) == 1
    assert data["claims"][0] == "Apollo 11 landed on the Moon."
    assert llm.calls == 1
