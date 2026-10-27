from contracts.schemas import AnswerDraft
from src.claims.extractor import (
    AtomicClaimExtractor,
    ClaimExtractionError,
)
from src.core.budget import BudgetConfig, BudgetGuard
from src.core.llm_client import OpenRouterLLMClient
from src.core.retry import RetryConfig, RetryPolicy


# Five cases covering claim extraction quality.
CASES = [
    (
        "multi_hop",
        "Christopher Nolan",
        "Christopher Nolan directed Inception. "
        "Inception stars Leonardo DiCaprio, "
        "who won an Academy Award in 2016.",
    ),
    (
        "false_claim",
        "Albert Einstein",
        "Albert Einstein was born in Paris in 1905 "
        "and won the Nobel Prize in Physics in 1925.",
    ),
    (
        "entity_date_number",
        "Apollo 11",
        "Apollo 11 landed on the Moon on July 20, 1969. "
        "Neil Armstrong and Buzz Aldrin walked on "
        "the lunar surface, while Michael Collins "
        "remained in lunar orbit.",
    ),
    (
        "complex_sentence",
        "Marie Curie",
        "Marie Curie, who was born in Warsaw in 1867, "
        "moved to Paris and received the Nobel Prize "
        "in Physics in 1903 and the Nobel Prize "
        "in Chemistry in 1911.",
    ),
    (
        "semantic_duplicates",
        "Paris",
        "Paris is the capital of France. "
        "The capital city of France is Paris. "
        "Paris is located in Europe.",
    ),
]


def main():
    # One API attempt per case; no automatic retries.
    guard = BudgetGuard(
        BudgetConfig(
            max_calls=5,
            max_cost_usd=0.02,
            min_key_remaining_usd=10.0,
        )
    )

    client = OpenRouterLLMClient(
        budget_guard=guard,
        retry_policy=RetryPolicy(
            RetryConfig(max_attempts=1)
        ),
    )

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
            print(f"FAILED: {exc}")
            break
        except Exception as exc:
            print(
                f"STOPPED: {type(exc).__name__}: {exc}"
            )
            break

        for claim in claims:
            print(f"{claim.claim_id}: {claim.claim_text}")

        print(f"CLAIM COUNT: {len(claims)}")

    print(f"\n{'=' * 55}")
    print("BUDGET SUMMARY")
    print(f"API attempts: {guard.calls}")
    print(f"Budget spent USD: {guard.spent_usd:.6f}")
    print(f"Reserved USD: {guard.reserved_usd:.6f}")


if __name__ == "__main__":
    main()
