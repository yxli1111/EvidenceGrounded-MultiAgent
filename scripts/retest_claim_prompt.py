from contracts.schemas import AnswerDraft
from src.claims.extractor import (
    AtomicClaimExtractor,
    ClaimExtractionError,
)
from src.core.budget import BudgetConfig, BudgetGuard
from src.core.llm_client import OpenRouterLLMClient
from src.core.retry import RetryConfig, RetryPolicy


# Two cases that previously showed coarse claim splitting.
CASES = [
    (
        "false_claim",
        "Albert Einstein",
        "Albert Einstein was born in Paris in 1905 "
        "and won the Nobel Prize in Physics in 1925.",
    ),
    (
        "complex_sentence",
        "Marie Curie",
        "Marie Curie, who was born in Warsaw in 1867, "
        "moved to Paris and received the Nobel Prize "
        "in Physics in 1903 and the Nobel Prize "
        "in Chemistry in 1911.",
    ),
]


def main():
    # Step 1: Limit API calls and estimated spending.
    guard = BudgetGuard(
        BudgetConfig(
            max_calls=2,
            max_cost_usd=0.01,
            min_key_remaining_usd=10.0,
        )
    )

    # Step 2: Use the real OpenRouter client.
    client = OpenRouterLLMClient(
        budget_guard=guard,
        retry_policy=RetryPolicy(
            RetryConfig(max_attempts=1)
        ),
    )

    # Step 3: Load the updated extraction prompt.
    extractor = AtomicClaimExtractor(
        model="openai/gpt-4o-mini",
        temperature=0,
        max_tokens=768,
    )

    for case_id, short_answer, explanation in CASES:
        draft = AnswerDraft(
            draft_id=case_id,
            short_answer=short_answer,
            explanation=explanation,
        )

        print(f"\n{'=' * 55}")
        print(f"CASE: {case_id}")
        print(f"INPUT: {explanation}")
        print("-" * 55)

        try:
            claims = extractor.extract_claims(
                draft=draft,
                llm=client,
            )
        except ClaimExtractionError as exc:
            print(f"EXTRACTION FAILED: {exc}")
            break
        except Exception as exc:
            print(
                f"STOPPED: {type(exc).__name__}: {exc}"
            )
            break

        for claim in claims:
            print(f"{claim.claim_id}: {claim.claim_text}")

        print(f"CLAIM COUNT: {len(claims)}")

    # Step 4: Report budget usage.
    print(f"\n{'=' * 55}")
    print("BUDGET SUMMARY")
    print(f"API attempts: {guard.calls}")
    print(f"Budget spent USD: {guard.spent_usd:.6f}")
    print(f"Reserved USD: {guard.reserved_usd:.6f}")


if __name__ == "__main__":
    main()
