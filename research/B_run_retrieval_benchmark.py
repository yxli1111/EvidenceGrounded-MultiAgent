
import csv
import time

from src.data.hotpot_loader import load_hotpotqa
from src.retrieval.retriever import EvidenceRetriever
from research.B_retrieval_diagnostics import (
    recall_at_k,
    full_coverage_at_k,
)


def main():
    inputs, gold_lookup = load_hotpotqa(limit=50)
    retriever = EvidenceRetriever()

    ks = [1, 3, 5, 10, 15, 20, 30]
    records = []

    for i, example in enumerate(inputs, start=1):
        start = time.perf_counter()

        results = retriever.retrieve(
            query=example["question"],
            context=example["context"],
            top_k=max(ks),
        )

        elapsed = time.perf_counter() - start

        gold = gold_lookup[example["example_id"]]
        gold_facts = gold["gold_supporting_facts"]

        record = {
            "example_id": example["example_id"],
            "question_type": example["question_type"],
            "retrieval_seconds": elapsed,
        }

        for k in ks:
            record[f"recall@{k}"] = recall_at_k(
                results, gold_facts, k
            )
            record[f"full_coverage@{k}"] = int(
                full_coverage_at_k(
                    results, gold_facts, k
                )
            )

        records.append(record)

        if i % 10 == 0:
            print(f"Processed {i}/{len(inputs)}")

    print("\n=== Retrieval Benchmark ===")

    for k in ks:
        mean_recall = sum(
            r[f"recall@{k}"] for r in records
        ) / len(records)

        mean_coverage = sum(
            r[f"full_coverage@{k}"] for r in records
        ) / len(records)

        print(
            f"k={k:2d} | "
            f"Recall={mean_recall:.3f} | "
            f"Full Coverage={mean_coverage:.3f}"
        )

    with open(
        "research/B_retrieval_results.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=records[0].keys()
        )
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
