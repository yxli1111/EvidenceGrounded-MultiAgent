
import csv
from pathlib import Path

from src.data.hotpot_loader import load_hotpotqa
from src.retrieval.retriever import EvidenceRetriever


BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "B_retrieval_results.csv"

MAX_CASES = 3
TOP_K = 10


def find_failed_bridge_ids():
    """Find bridge questions with incomplete gold evidence."""

    failed_ids = []

    with RESULTS_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                row["question_type"] == "bridge"
                and float(row["full_coverage@10"]) == 0
            ):
                failed_ids.append(row["example_id"])

    return failed_ids[:MAX_CASES]


def main():
    failed_ids = find_failed_bridge_ids()

    if not failed_ids:
        print("No failed bridge examples found.")
        return

    # Load the same 50-example diagnostic subset.
    inputs, gold_lookup = load_hotpotqa(limit=50)

    examples_by_id = {
        example["example_id"]: example
        for example in inputs
    }

    retriever = EvidenceRetriever()

    for example_id in failed_ids:
        example = examples_by_id[example_id]
        gold = gold_lookup[example_id]

        gold_keys = {
            (fact["title"], fact["sentence_id"])
            for fact in gold["gold_supporting_facts"]
        }

        print("\n" + "=" * 70)
        print("QUESTION:", example["question"])
        print("GOLD ANSWER:", gold["gold_answer"])
        print("=" * 70)

        print("\nGOLD SUPPORTING FACTS:")

        for sentence in example["context"]:
            key = (
                sentence["title"],
                sentence["sentence_id"]
            )

            if key in gold_keys:
                print(
                    f"[{sentence['title']} "
                    f"#{sentence['sentence_id']}]"
                )
                print(sentence["text"])

        results = retriever.retrieve(
            query=example["question"],
            context=example["context"],
            top_k=TOP_K
        )
        
        expanded_results = retriever.retrieve_with_neighbors(
            query=example["question"],
            context=example["context"],
            total_k=5,
            seed_k=3,
            window=1,
        )

        print("\nADJACENT SENTENCE EXPANSION (K=5):")

        expanded_keys = set()

        for item in expanded_results:
            key = (item["doc_title"], item["sentence_id"])
            expanded_keys.add(key)

            marker = "[GOLD]" if key in gold_keys else "[    ]"

            print(
                f"{marker} Rank {item['rank']}: "
                f"{item['doc_title']} [{item['sentence_id']}] "
                f"source={item['selection_source']}"
            )

        print(
            "Expanded Recall@5:",
            len(expanded_keys & gold_keys) / len(gold_keys)
        )
        print(
            "Expanded Full Coverage@5:",
            int(gold_keys.issubset(expanded_keys))
        )


        retrieved_keys = {
            (item["doc_title"], item["sentence_id"])
            for item in results
        }

        print("\nTOP-10 RETRIEVED EVIDENCE:")

        for item in results:
            key = (
                item["doc_title"],
                item["sentence_id"]
            )

            marker = (
                "[GOLD]" if key in gold_keys
                else "[    ]"
            )

            print(
                f"{marker} Rank {item['rank']}: "
                f"{item['doc_title']} "
                f"[{item['sentence_id']}] "
                f"score={item['score']:.4f}"
            )
            print(item["text"])

        missing = gold_keys - retrieved_keys

        print("\nMISSING GOLD EVIDENCE:")

        for title, sentence_id in sorted(missing):
            print(f"  {title} [{sentence_id}]")


if __name__ == "__main__":
    main()
