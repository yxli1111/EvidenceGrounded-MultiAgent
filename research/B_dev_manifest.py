
import json
from pathlib import Path

from src.data.hotpot_loader import load_hotpotqa


OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "B_dev_50_ids.json"
)


def main():
    inputs, _ = load_hotpotqa(
        split="validation",
        limit=50,
    )

    example_ids = [
        example["example_id"]
        for example in inputs
    ]

    if len(example_ids) != 50:
        raise ValueError("Expected exactly 50 examples")

    if len(set(example_ids)) != 50:
        raise ValueError("Duplicate example IDs detected")

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "dataset": "hotpotqa/hotpot_qa",
                "configuration": "distractor",
                "split": "validation",
                "selection": "first_50",
                "purpose": "development_only",
                "example_ids": example_ids,
            },
            file,
            indent=2,
        )

    print(f"Saved {len(example_ids)} IDs")
    print(f"Manifest: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
