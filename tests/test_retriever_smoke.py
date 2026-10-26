
from src.data.hotpot_loader import load_hotpotqa
from src.retrieval.retriever import EvidenceRetriever
from research.B_retrieval_diagnostics import (
    recall_at_k,
    full_coverage_at_k,
)


def print_results(name, results, gold_facts, k=5):
    """Display retrieval results and evaluate gold evidence coverage."""

    gold_set = {
        (fact["title"], fact["sentence_id"])
        for fact in gold_facts
    }

    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)

    for item in results:
        key = (item["doc_title"], item["sentence_id"])
        is_gold = key in gold_set

        marker = "[GOLD]" if is_gold else "[    ]"

        print(
            f"{marker} Rank {item['rank']}: "
            f"{item['doc_title']} "
            f"[{item['sentence_id']}] "
            f"score={item['score']:.4f}"
        )
        print(item["text"])
        print()

    recall = recall_at_k(results, gold_facts, k)
    coverage = full_coverage_at_k(results, gold_facts, k)

    print(f"Recall@{k}: {recall:.3f}")
    print(f"Full Coverage@{k}: {int(coverage)}")

    return recall, coverage


def main():
    # Load one HotpotQA example.
    inputs, gold_lookup = load_hotpotqa(limit=1)
    example = inputs[0]

    gold_facts = gold_lookup[
        example["example_id"]
    ]["gold_supporting_facts"]

    # Initialize the embedding retriever once.
    retriever = EvidenceRetriever()

    print("Question:", example["question"])
    print("Gold Supporting Facts:")

    for fact in gold_facts:
        print(
            f"  {fact['title']} [{fact['sentence_id']}]"
        )

    # -----------------------------------------
    # Method 1: Single-Query Retrieval
    # -----------------------------------------
    single_results = retriever.retrieve(
        query=example["question"],
        context=example["context"],
        top_k=5,
    )

    single_recall, single_coverage = print_results(
        name="Single-Query Retrieval",
        results=single_results,
        gold_facts=gold_facts,
        k=5,
    )

    # -----------------------------------------
    # Method 2: Multi-Query Retrieval
    # -----------------------------------------
    # Manually written subqueries for this smoke test.
    # NOT suitable as final experimental results.
    queries = [
        "Scott Derrickson nationality",
        "Ed Wood nationality",
    ]

    multi_results = retriever.retrieve_multi_query(
        queries=queries,
        context=example["context"],
        total_k=5,
    )

    multi_recall, multi_coverage = print_results(
        name="Multi-Query Retrieval",
        results=multi_results,
        gold_facts=gold_facts,
        k=5,
    )

    # -----------------------------------------
    # Summary
    # -----------------------------------------
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)

    print(
        f"{'Metric':<22}"
        f"{'Single-Query':<16}"
        f"{'Multi-Query':<16}"
    )

    print(
        f"{'Recall@5':<22}"
        f"{single_recall:<16.3f}"
        f"{multi_recall:<16.3f}"
    )

    print(
        f"{'Full Coverage@5':<22}"
        f"{int(single_coverage):<16}"
        f"{int(multi_coverage):<16}"
    )


if __name__ == "__main__":
    main()
