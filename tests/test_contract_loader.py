
from contracts.schemas import QAInput, QAGold, Sentence
from src.data.contract_loader import load_hotpotqa_contract


def main():
    inputs, gold_lookup = load_hotpotqa_contract(
        split="validation",
        limit=1,
    )

    assert len(inputs) == 1

    example = inputs[0]

    assert isinstance(example, QAInput)
    assert isinstance(example.context[0], Sentence)

    gold = gold_lookup[example.example_id]

    assert isinstance(gold, QAGold)
    assert gold.example_id == example.example_id

    print("Question:", example.question)
    print("Context sentences:", len(example.context))
    print("Gold answer:", gold.gold_answer)

    print("Contract Loader smoke test passed.")


if __name__ == "__main__":
    main()
