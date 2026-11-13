import os

from dotenv import load_dotenv

from contracts.schemas import AnswerDraft
from src.claims.extractor import (
    AtomicClaimExtractor,
    ClaimExtractionError,
)
from src.core.budget import BudgetConfig, BudgetGuard
from src.core.llm_client import OpenRouterLLMClient
from src.core.retry import RetryConfig, RetryPolicy


def main():
    load_dotenv()

    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is missing")

    # Limit the smoke test to three real API attempts.
    guard = BudgetGuard(
        BudgetConfig(
            max_calls=3,
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
        max_tokens=512,
    )

    examples = [
        AnswerDraft(
            draft_id="smoke-1",
            short_answer="Marie Curie",
            explanation=(
                "Marie Curie was born in Warsaw in 1867 "
                "and won two Nobel Prizes."
            ),
        ),
        AnswerDraft(
            draft_id="smoke-2",
            short_answer="Albert Einstein",
            explanation=(
                "Albert Einstein was born in Germany. "
                "He received the Nobel Prize in Physics "
                "in 1921."
            ),
        ),
        AnswerDraft(
            draft_id="smoke-3",
            short_answer="Unknown",
            explanation=(
                "What a wonderful day! "
                "I think this is amazing."
            ),
        ),
    ]

    for example in examples:
        print(f"\n=== {example.draft_id} ===")

        try:
            claims = extractor.extract_claims(
                draft=example,
                llm=client,
            )

        except ClaimExtractionError as exc:
            print(f"Extraction failed: {exc}")
            break

        for claim in claims:
            print(
                f"{claim.claim_id}: {claim.claim_text}"
            )

        print(f"Claim count: {len(claims)}")

    print("\n=== Budget summary ===")
    print(f"API attempts: {guard.calls}")
    print(f"Budget spent (USD): {guard.spent_usd:.6f}")
    print(f"Budget reserved (USD): {guard.reserved_usd:.6f}")


if __name__ == "__main__":
    main()
