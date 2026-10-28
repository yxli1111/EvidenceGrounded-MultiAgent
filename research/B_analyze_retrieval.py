
import csv
from collections import defaultdict
from pathlib import Path


RESULTS_PATH = (
    Path(__file__).resolve().parent
    / "B_retrieval_results.csv"
)

KS = [1, 3, 5, 10, 15, 20, 30]


def main():
    groups = defaultdict(list)

    with RESULTS_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            groups[row["question_type"]].append(row)

    print("\n=== Retrieval by Question Type ===")

    for question_type, records in sorted(groups.items()):
        print(
            f"\nQuestion type: {question_type}"
            f" | n={len(records)}"
        )

        for k in KS:
            recall = sum(
                float(r[f"recall@{k}"])
                for r in records
            ) / len(records)

            coverage = sum(
                float(r[f"full_coverage@{k}"])
                for r in records
            ) / len(records)

            print(
                f"k={k:2d} | "
                f"Recall={recall:.3f} | "
                f"Full Coverage={coverage:.3f}"
            )


if __name__ == "__main__":
    main()
