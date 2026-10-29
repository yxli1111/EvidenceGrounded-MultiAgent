
from datasets import load_dataset


def normalize_example(row: dict) -> tuple[dict, dict]:
    """Convert one HotpotQA row into separate input and gold records."""

    context = []

    for title, sentences in zip(
        row["context"]["title"],
        row["context"]["sentences"]
    ):
        for sentence_id, text in enumerate(sentences):
            context.append({
                "title": title,
                "sentence_id": sentence_id,
                "text": text
            })

    qa_input = {
        "example_id": row["id"],
        "question": row["question"],
        "question_type": row["type"],
        "context": context
    }

    gold = {
        "example_id": row["id"],
        "gold_answer": row["answer"],
        "gold_supporting_facts": [
            {
                "title": title,
                "sentence_id": sentence_id
            }
            for title, sentence_id in zip(
                row["supporting_facts"]["title"],
                row["supporting_facts"]["sent_id"]
            )
        ]
    }

    return qa_input, gold


def load_hotpotqa(
    split: str = "validation",
    limit: int | None = None
) -> tuple[list[dict], dict[str, dict]]:
    """Load HotpotQA distractor examples with gold kept separately."""

    split_spec = split if limit is None else f"{split}[:{limit}]"

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split=split_spec
    )

    inputs = []
    gold_lookup = {}

    for row in dataset:
        qa_input, gold = normalize_example(row)
        inputs.append(qa_input)
        gold_lookup[qa_input["example_id"]] = gold

    return inputs, gold_lookup


if __name__ == "__main__":
    inputs, gold = load_hotpotqa(limit=3)

    print(f"Loaded {len(inputs)} examples")

    example = inputs[0]
    print("Question:", example["question"])
    print("Context sentences:", len(example["context"]))
    print("Gold answer:", gold[example["example_id"]]["gold_answer"])
