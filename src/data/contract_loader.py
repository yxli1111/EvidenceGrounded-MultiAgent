
"""Contract-compatible HotpotQA dataset loader."""

from contracts.schemas import QAInput, QAGold, Sentence
from src.data.hotpot_loader import load_hotpotqa


def load_hotpotqa_contract(
    split: str = "validation",
    limit: int = 50,
) -> tuple[list[QAInput], dict[str, QAGold]]:
    """Load HotpotQA and convert records to shared Contract dataclasses."""

    raw_inputs, raw_gold_lookup = load_hotpotqa(
        split=split,
        limit=limit,
    )

    inputs = []

    for example in raw_inputs:
        sentences = [
            Sentence(
                title=sentence["title"],
                sentence_id=sentence["sentence_id"],
                text=sentence["text"],
            )
            for sentence in example["context"]
        ]

        inputs.append(
            QAInput(
                example_id=example["example_id"],
                question=example["question"],
                context=sentences,
                question_type=example.get("question_type"),
            )
        )

    gold_lookup = {}

    for example_id, gold in raw_gold_lookup.items():
        gold_lookup[example_id] = QAGold(
            example_id=example_id,
            gold_answer=gold["gold_answer"],
            gold_supporting_facts=[
                (fact["title"], fact["sentence_id"])
                for fact in gold["gold_supporting_facts"]
            ],
        )

    return inputs, gold_lookup
