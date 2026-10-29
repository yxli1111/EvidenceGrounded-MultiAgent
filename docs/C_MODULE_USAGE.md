# Contributor C — Module Usage Guide

**Project:** EvidenceGrounded-MultiAgent
**Module:** Shared LLM Infrastructure & Atomic Claim Extraction
**Status:** Implementation complete; team integration pending
**Python:** 3.11.3

## 1. Overview

Contributor C provides two core services:

1. **Shared LLM Infrastructure:** A unified OpenRouter client with API usage tracking, budget protection, persistent caching, bounded retries, and structured error handling.
2. **Atomic Claim Extraction:** An LLM-based service that decomposes an answer explanation into independently verifiable factual claims.

Both services follow the public interfaces defined in `contracts/interfaces.py` and the data structures in `contracts/schemas.py`.

## 2. Installation

Create and activate a Python virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The C module has been tested with:

- Python 3.11.3
- openai 3.37.0
- python-dotenv 1.2.4
- httpx 0.28.1
- pytest 9.1.1

The final team-wide dependency list may require additional packages.

## 3. API Key Configuration

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Never commit the `.env` file or real API credentials to GitHub.

Ensure `.gitignore` contains:

```gitignore
.env
.venv/
/logs/
```

The OpenRouter client reads the API key from the environment.

## 4. OpenRouter LLM Client

The shared implementation is located at:

`src/core/llm_client.py`

Basic usage:

```python
from src.core.llm_client import OpenRouterLLMClient

llm = OpenRouterLLMClient()

response = llm.generate(
    stage="answer_generation",
    messages=[
        {
            "role": "user",
            "content": "Who discovered penicillin?"
        }
    ],
    model="openai/gpt-4o-mini",
    temperature=0,
    max_tokens=128,
)

print(response.text)
print(response.usage)
```

The `generate()` method returns an `LLMResponse` containing generated text and an `LLMUsage` record.

Usage metadata can include input tokens, output tokens, estimated or reported cost, latency, and cache-hit status.

## 5. Atomic Claim Extraction

The implementation is located at:

`src/claims/extractor.py`

The extraction prompt is stored in:

`prompts/claim_extraction/system.txt`

Example:

```python
from contracts.schemas import AnswerDraft
from src.claims.extractor import AtomicClaimExtractor
from src.core.llm_client import OpenRouterLLMClient

llm = OpenRouterLLMClient()
extractor = AtomicClaimExtractor()

draft = AnswerDraft(
    draft_id="example-1",
    short_answer="Marie Curie",
    explanation=(
        "Marie Curie was born in Warsaw in 1867 "
        "and won two Nobel Prizes."
    ),
)

claims = extractor.extract_claims(
    draft=draft,
    llm=llm,
)

for claim in claims:
    print(claim.claim_id, claim.claim_text)
```

Illustrative output:

```text
c1 Marie Curie was born in Warsaw.
c2 Marie Curie was born in 1867.
c3 Marie Curie won two Nobel Prizes.
```

The extraction service returns `list[AtomicClaim]`.

Each `AtomicClaim` contains:

- `claim_id`: Identifier assigned within the extraction result.
- `claim_text`: A self-contained factual statement.
- `source_span`: Optional character offsets; currently `None`.

The extractor uses `draft.explanation` as its primary source. The short answer is provided only as interpretation context.

## 6. Budget Protection

The shared budget implementation is located at:

`src/core/budget.py`

Example configuration:

```python
from src.core.budget import BudgetConfig, BudgetGuard
from src.core.llm_client import OpenRouterLLMClient

guard = BudgetGuard(
    BudgetConfig(
        max_calls=5,
        max_cost_usd=0.02,
        min_key_remaining_usd=10.0,
    )
)

llm = OpenRouterLLMClient(
    budget_guard=guard
)
```

The budget guard checks estimated costs and available key balance before permitting requests.

Budget limits are based on estimates and internal accounting; they are not a guaranteed hard cap on provider billing.

## 7. Persistent Cache

The SQLite cache implementation is located at:

`src/core/cache.py`

The cache is optional and must be explicitly configured.

Cache keys account for request parameters, including model and messages.

Successful cached responses can avoid repeated API calls and reduce costs.

Cache-hit usage records indicate that no new model request was made.

The cache must not be treated as a substitute for checking whether a model or prompt configuration has changed.

## 8. Retry Policy

The retry implementation is located at:

`src/core/retry.py`

Example:

```python
from src.core.retry import RetryConfig, RetryPolicy

retry_policy = RetryPolicy(
    RetryConfig(
        max_attempts=2,
        backoff_seconds=0.5,
        max_retry_after_seconds=30.0,
    )
)
```

The policy supports bounded retries for transient network failures and retryable HTTP status codes, including 429 and selected 5xx responses.

When a valid `Retry-After` header is available, the policy uses it subject to the configured waiting-time limit.

Permanent authentication, insufficient-credit, and invalid-request errors are not blindly retried.

## 9. Error Handling

The shared client defines:

- `LLMClientError`: Base LLM service error.
- `LLMAuthenticationError`: Authentication or permission error.
- `LLMInsufficientCreditError`: Insufficient credit.
- `LLMRequestError`: Invalid model or request.

The claim extractor defines:

- `ClaimExtractionError`: Claim parsing, validation, or repair failure.

Example:

```python
from src.core.llm_client import LLMClientError
from src.claims.extractor import ClaimExtractionError

try:
    claims = extractor.extract_claims(draft, llm)

except LLMClientError as exc:
    print("LLM service failed:", type(exc).__name__)

except ClaimExtractionError as exc:
    print("Claim extraction failed:", type(exc).__name__)
```

A genuine zero-claim result returns `[]`.

API, parsing, or validation failures must not be silently represented as empty claims.

Malformed extraction output may trigger one additional LLM repair request, subject to the same API budget controls.

## 10. Testing

Run all available tests:

```powershell
python -m pytest tests -q
```

Last locally verified result:

```text
98 passed
```

Claim Extractor tests:

```powershell
python -m pytest tests/test_claim_extractor.py -v
```

Error-handling tests:

```powershell
python -m pytest tests/test_llm_errors.py -v
```

These tests use mocks and do not require real OpenRouter API calls.

Optional live smoke tests are available under `scripts/`. Live tests consume API credits and should be run only with explicit budget limits.

## 11. Known Limitations

1. Claim extraction quality depends on the selected model and prompt.
2. Semantic deduplication is primarily LLM-guided; Python performs conservative textual deduplication.
3. `source_span` is currently returned as `None`.
4. Claim IDs such as `c1` and `c2` are local to each extraction and should be associated with the corresponding draft or example ID.
5. JSON repair introduces additional API calls and costs when triggered.
6. Budget checks use estimated costs and cannot guarantee an absolute provider-side spending ceiling.
7. End-to-end integration with the other contributors' modules remains pending.

## 12. Integration Notes

The expected claim-level workflow is:

```text
AnswerDraft
    |
    v
AtomicClaimExtractor
    |
    v
list[AtomicClaim]
    |
    v
Evidence Retrieval
    |
    v
Claim-Level Verification
    |
    v
Experiment Runner / Evaluation
```

The experiment runner should catch LLM and extraction errors and record failures explicitly rather than forwarding error messages as claims.

The final public Contract v1.0 should be confirmed before merging this module into the shared main branch.
