
def recall_at_k(
    retrieved: list[dict],
    gold_facts: list[dict],
    k: int
) -> float:
    gold = {
        (item["title"], item["sentence_id"])
        for item in gold_facts
    }

    if not gold:
        return 0.0

    predicted = {
        (item["doc_title"], item["sentence_id"])
        for item in retrieved[:k]
    }

    return len(gold & predicted) / len(gold)


def full_coverage_at_k(
    retrieved: list[dict],
    gold_facts: list[dict],
    k: int
) -> bool:
    gold = {
        (item["title"], item["sentence_id"])
        for item in gold_facts
    }

    if not gold:
        return False

    predicted = {
        (item["doc_title"], item["sentence_id"])
        for item in retrieved[:k]
    }

    return gold.issubset(predicted)
