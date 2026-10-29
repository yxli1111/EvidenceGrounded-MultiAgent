# Contributor C — Handoff Document

**Project:** EvidenceGrounded-MultiAgent
**Contributor:** C
**Branch:** `feature/cmj-claims-openrouter`
**Status:** Ready for code review; team integration pending
**Python Version:** 3.11.3

## 1. Scope of Delivery

Contributor C is responsible for:

- Shared OpenRouter LLM infrastructure.
- API usage, token, cost, and latency tracking.
- API budget protection and remaining-credit checks.
- SQLite-based response caching.
- Bounded retries and HTTP `Retry-After` handling.
- Structured LLM error classification.
- Atomic factual claim extraction.
- Claim output validation, deduplication, and bounded JSON repair.

The implementation follows the provisional public contracts. Final compatibility must be checked against the approved Contract v1.0.

## 2. Delivered Files

### Core LLM Infrastructure

| File | Responsibility |
|---|---|
| `src/core/llm_client.py` | Unified OpenRouter client |
| `src/core/budget.py` | API budget management |
| `src/core/cache.py` | Persistent SQLite cache |
| `src/core/cost_estimator.py` | Pre-request cost estimation |
| `src/core/key_usage.py` | OpenRouter key usage information |
| `src/core/retry.py` | Bounded retry policy |
| `src/core/usage.py` | Success and failure usage logging |

### Claim Extraction

| File | Responsibility |
|---|---|
| `src/claims/extractor.py` | Atomic Claim Extractor |
| `prompts/claim_extraction/system.txt` | Claim extraction prompt |

### Tests and Documentation

- `tests/test_llm_client.py`
- `tests/test_llm_errors.py`
- `tests/test_llm_integration.py`
- `tests/test_budget.py`
- `tests/test_cache.py`
- `tests/test_cost_estimator.py`
- `tests/test_key_usage.py`
- `tests/test_retry.py`
- `tests/test_usage.py`
- `tests/test_claim_extractor.py`
- `docs/C_MODULE_USAGE.md`
- `requirements.txt`

Development and smoke-test scripts are available under `scripts/`.

## 3. Public Interfaces

### 3.1 LLMClient.generate()

```python
def generate(
    self,
    *,
    stage: str,
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
) -> LLMResponse:
    ...
```

The client returns an `LLMResponse` containing the generated text and an `LLMUsage` record.

Example:

```python
from src.core.llm_client import OpenRouterLLMClient

llm = OpenRouterLLMClient()

response = llm.generate(
    stage="answer_generation",
    messages=[
        {"role": "user", "content": "Hello"}
    ],
    model="openai/gpt-4o-mini",
    temperature=0,
    max_tokens=128,
)
```

### 3.2 ClaimPort.extract_claims()

```python
def extract_claims(
    self,
    draft: AnswerDraft,
    llm: LLMClient,
) -> list[AtomicClaim]:
    ...
```

Example:

```python
from contracts.schemas import AnswerDraft
from src.claims.extractor import AtomicClaimExtractor

draft = AnswerDraft(
    draft_id="example-1",
    short_answer="Marie Curie",
    explanation=(
        "Marie Curie was born in Warsaw in 1867."
    ),
)

extractor = AtomicClaimExtractor()
claims = extractor.extract_claims(draft, llm)
```

Each returned claim contains:

- `claim_id`: Local identifier, such as `c1`.
- `claim_text`: Independently verifiable factual statement.
- `source_span`: Currently `None`.

Claim IDs should be associated with the corresponding draft or example ID to avoid collisions across examples.

## 4. Error Handling Contract

The LLM client defines the following exception hierarchy:

```text
LLMClientError
├── LLMAuthenticationError
├── LLMInsufficientCreditError
└── LLMRequestError
```

The claim extraction service defines `ClaimExtractionError` separately.

Expected behavior:

| Situation | Result |
|---|---|
| Successful API call | Return `LLMResponse` |
| Successful extraction | Return `list[AtomicClaim]` |
| Genuine zero-claim result | Return `[]` |
| Authentication or permission failure | Raise `LLMAuthenticationError` |
| Insufficient credit | Raise `LLMInsufficientCreditError` |
| Invalid API request | Raise `LLMRequestError` |
| Retry exhaustion | Raise `LLMClientError` |
| Invalid claim output after repair | Raise `ClaimExtractionError` |

LLM client errors propagate through the Claim Extractor without being converted into parsing errors.

Malformed claim output may trigger one additional LLM repair request. The repair request uses the same budget-controlled client.

Errors must never be represented as ordinary claims or silently converted into empty results.

## 5. Budget, Cache, and Retry

### Budget

The `BudgetGuard` supports:

- Maximum API attempt count.
- Maximum estimated spending.
- Minimum required key remaining balance.
- Cost reservation and settlement.

Budget limits are local safeguards and do not guarantee an absolute provider-side billing limit.

### Cache

The optional SQLite cache stores successful LLM responses and can prevent repeated API calls.

Cache keys incorporate request parameters to distinguish different model and prompt configurations.

### Retry

The `RetryPolicy` supports bounded retries for temporary network errors and selected HTTP status codes.

Valid `Retry-After` headers are respected subject to a configurable maximum waiting period.

Non-retryable authentication, credit, and invalid-request errors are surfaced immediately.

## 6. Validation Results

### Automated Tests

Last verified command:

```powershell
python -m pytest tests -q
```

Result:

```text
98 passed
```

This includes unit and mocked integration tests for LLM infrastructure, budget protection, caching, retries, error handling, and claim extraction.

### Real OpenRouter Smoke Tests

The following scenarios were manually checked:

- Basic OpenRouter API connectivity.
- Simple atomic claim decomposition.
- Pronoun resolution.
- Genuine zero-claim output.
- Multi-hop factual statements.
- Preservation of false claims.
- Entity, date, and number preservation.
- Complex sentence decomposition.
- Semantic duplicate removal.
- Prompt refinement for finer atomicity.

These tests demonstrated expected behavior on a small development sample. They do not establish dataset-wide extraction accuracy.

## 7. Integration Requirements

### Contributor A — Answer Generation

Contributor A produces `AnswerDraft` objects.

The Claim Extractor uses:

- `draft.explanation` as the factual extraction source.
- `draft.short_answer` as interpretation context.

### Contributor B — Evidence Retrieval

Contributor B can retrieve evidence using each claim's `claim_text`.

Evidence should be associated with the relevant claim ID and example ID.

Contributor C does not implement evidence retrieval.

### Contributor D — Verification

Contributor D receives claims and their retrieved evidence through the experiment pipeline.

Contributor C extracts factual statements without verifying or correcting them.

### Contributor E — Runner and Evaluation

Contributor E is responsible for integrating the modules and handling exceptions.

Expected flow:

```text
AnswerDraft
    ↓
Claim Extractor
    ↓
list[AtomicClaim]
    ↓
Evidence Retrieval
    ↓
Claim-Level Verification
    ↓
Runner / Evaluation
```

The Runner should distinguish successful empty extraction from API or parsing failure.

On failure, the Runner should record an error status in the experiment record rather than passing an error message to the retrieval module.

## 8. Known Limitations

1. Claim extraction quality depends on the selected LLM and prompt.
2. Semantic deduplication relies partly on LLM behavior.
3. `source_span` is currently `None`.
4. Claim IDs are local to each extraction result.
5. JSON repair can require an additional paid API request.
6. Budget limits are estimate-based safeguards.
7. End-to-end integration with A/B/D/E is not yet verified.
8. Final Contract v1.0 compatibility is pending team confirmation.

## 9. Pending Team Coordination

Before final merge:

- Confirm the approved Contract v1.0.
- Ensure public schemas and interfaces are provided by the shared repository.
- Verify that the Runner uses the shared LLMClient.
- Test Claim Extractor output with Contributor B's retrieval module.
- Confirm Contributor D's expected claim-level input.
- Verify that Contributor E records extraction failures correctly.
- Consolidate team-wide dependencies.
- Run the full test suite after integration.

## 10. Research Responsibility

Contributor C is responsible for RQ2, comparing M4 and M5 to study the effect of atomic claim decomposition.

The research phase will include:

- M4 versus M5 comparisons under controlled conditions.
- Analysis of claim extraction omissions and over-decomposition.
- Additional token and cost overhead.
- Claim-level verification outcomes.
- Failure case analysis.

The research notebook is planned as:

`research/C_claim_ablation.ipynb`

This work depends on the integrated experiment pipeline and evaluation outputs from Contributor E.

## 11. Review Checklist

- [x] Shared LLM infrastructure implemented.
- [x] Atomic Claim Extractor implemented.
- [x] Unit and mocked integration tests passing.
- [x] Basic real API smoke tests completed.
- [x] Usage documentation prepared.
- [ ] Approved Contract v1.0 synchronized.
- [ ] Cross-contributor integration completed.
- [ ] Team-wide dependency configuration finalized.
- [ ] RQ2 experiments completed.
- [ ] Final PR review and merge completed.
