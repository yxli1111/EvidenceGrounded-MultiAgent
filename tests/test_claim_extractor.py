from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from contracts.schemas import AnswerDraft
from src.claims.extractor import (
    AtomicClaimExtractor,
    ClaimExtractionError,
)
from src.core.llm_client import (
    LLMClientError,
    LLMInsufficientCreditError,
)


def make_draft(explanation: str):
    return AnswerDraft(
        draft_id="draft-1",
        short_answer="Marie Curie",
        explanation=explanation,
    )


def make_llm_response(text: str):
    llm = MagicMock()
    llm.generate.return_value = SimpleNamespace(
        text=text
    )
    return llm


def test_extract_multiple_claims():
    extractor = AtomicClaimExtractor()

    draft = make_draft(
        "Marie Curie was born in Warsaw in 1867 "
        "and won two Nobel Prizes."
    )

    llm = make_llm_response(
        '{"claims": ['
        '"Marie Curie was born in Warsaw.",'
        '"Marie Curie was born in 1867.",'
        '"Marie Curie won two Nobel Prizes."'
        ']}'
    )

    claims = extractor.extract_claims(draft, llm)

    assert len(claims) == 3
    assert [c.claim_id for c in claims] == [
        "c1", "c2", "c3"
    ]
    assert claims[0].claim_text == (
        "Marie Curie was born in Warsaw."
    )
    assert claims[0].source_span is None

    llm.generate.assert_called_once()
    assert (
        llm.generate.call_args.kwargs["stage"]
        == "claim_extraction"
    )


def test_empty_explanation():
    extractor = AtomicClaimExtractor()
    draft = make_draft("   ")
    llm = MagicMock()

    assert extractor.extract_claims(draft, llm) == []
    llm.generate.assert_not_called()


def test_no_factual_claims():
    extractor = AtomicClaimExtractor()
    draft = make_draft("What a wonderful day!")
    llm = make_llm_response('{"claims": []}')

    assert extractor.extract_claims(draft, llm) == []
    llm.generate.assert_called_once()


def test_duplicate_claims_removed():
    extractor = AtomicClaimExtractor()
    draft = make_draft("Paris is in France.")

    llm = make_llm_response(
        '{"claims": ['
        '"Paris is in France.",'
        '"paris is in france.",'
        '"Paris is the capital of France."'
        ']}'
    )

    claims = extractor.extract_claims(draft, llm)

    assert len(claims) == 2
    assert claims[0].claim_id == "c1"
    assert claims[1].claim_id == "c2"


def test_invalid_json_raises_error():
    extractor = AtomicClaimExtractor(
        max_repair_attempts=0
    )
    draft = make_draft("Paris is in France.")
    llm = make_llm_response("Not valid JSON")

    with pytest.raises(ClaimExtractionError):
        extractor.extract_claims(draft, llm)


def test_invalid_claim_structure():
    extractor = AtomicClaimExtractor(
        max_repair_attempts=0
    )
    draft = make_draft("Paris is in France.")

    llm = make_llm_response(
        '{"claims": "Paris is in France."}'
    )

    with pytest.raises(ClaimExtractionError):
        extractor.extract_claims(draft, llm)


def test_invalid_claim_item():
    extractor = AtomicClaimExtractor(
        max_repair_attempts=0
    )
    draft = make_draft("Paris is in France.")

    llm = make_llm_response(
        '{"claims": [123]}'
    )

    with pytest.raises(ClaimExtractionError):
        extractor.extract_claims(draft, llm)


def test_llm_failure_raises_error():
    extractor = AtomicClaimExtractor()
    draft = make_draft("Paris is in France.")

    llm = MagicMock()
    llm.generate.side_effect = LLMClientError(
        "Mock API failure"
    )

    # Preserve the original LLMClientError.
    with pytest.raises(
        LLMClientError,
        match="Mock API failure"
    ):
        extractor.extract_claims(draft, llm)

    llm.generate.assert_called_once()


def test_missing_claims_key():
    extractor = AtomicClaimExtractor(
        max_repair_attempts=0
    )
    draft = make_draft("Paris is in France.")
    llm = make_llm_response('{"result": []}')

    with pytest.raises(ClaimExtractionError):
        extractor.extract_claims(draft, llm)


def test_json_repair_success():
    extractor = AtomicClaimExtractor()
    draft = make_draft("Paris is in France.")

    llm = MagicMock()
    llm.generate.side_effect = [
        SimpleNamespace(
            text='{"claims": ["Paris is in France."'
        ),
        SimpleNamespace(
            text='{"claims": ["Paris is in France."]}'
        ),
    ]

    claims = extractor.extract_claims(draft, llm)

    assert len(claims) == 1
    assert claims[0].claim_text == (
        "Paris is in France."
    )
    assert llm.generate.call_count == 2

    stages = [
        call.kwargs["stage"]
        for call in llm.generate.call_args_list
    ]

    assert stages == [
        "claim_extraction",
        "claim_extraction_repair",
    ]


def test_json_repair_failure():
    extractor = AtomicClaimExtractor()
    draft = make_draft("Paris is in France.")

    llm = MagicMock()
    llm.generate.side_effect = [
        SimpleNamespace(text="Invalid JSON"),
        SimpleNamespace(text="Still invalid"),
    ]

    with pytest.raises(
        ClaimExtractionError,
        match="JSON repair failed"
    ):
        extractor.extract_claims(draft, llm)

    assert llm.generate.call_count == 2


def test_json_repair_rejects_ambiguous_empty():
    extractor = AtomicClaimExtractor()
    draft = make_draft("Paris is in France.")

    llm = MagicMock()
    llm.generate.side_effect = [
        SimpleNamespace(text="Invalid JSON"),
        SimpleNamespace(text='{"claims": []}'),
    ]

    with pytest.raises(
        ClaimExtractionError,
        match="ambiguous empty"
    ):
        extractor.extract_claims(draft, llm)

    assert llm.generate.call_count == 2


def test_repair_api_failure_propagates():
    extractor = AtomicClaimExtractor()
    draft = make_draft("Paris is in France.")

    llm = MagicMock()
    llm.generate.side_effect = [
        SimpleNamespace(text="Invalid JSON"),
        LLMInsufficientCreditError(
            "Insufficient credit"
        ),
    ]

    with pytest.raises(
        LLMInsufficientCreditError
    ):
        extractor.extract_claims(draft, llm)

    assert llm.generate.call_count == 2


def test_json_repair_disabled():
    extractor = AtomicClaimExtractor(
        max_repair_attempts=0
    )
    draft = make_draft("Paris is in France.")
    llm = make_llm_response("Invalid JSON")

    with pytest.raises(ClaimExtractionError):
        extractor.extract_claims(draft, llm)

    llm.generate.assert_called_once()


def test_invalid_repair_configuration():
    with pytest.raises(ValueError):
        AtomicClaimExtractor(
            max_repair_attempts=2
        )


def test_unexpected_llm_failure():
    extractor = AtomicClaimExtractor()
    draft = make_draft("Paris is in France.")

    llm = MagicMock()
    llm.generate.side_effect = RuntimeError(
        "Unexpected mock failure"
    )

    with pytest.raises(
        ClaimExtractionError,
        match="Unexpected LLM extraction failure"
    ):
        extractor.extract_claims(draft, llm)

    llm.generate.assert_called_once()
