from contracts.schemas import Sentence, Evidence
from src.retrieval.contract_adapter import ContractEvidenceRetriever


def main():
    context = [
        Sentence(
            title="Scott Derrickson",
            sentence_id=0,
            text="Scott Derrickson is an American director.",
        ),
        Sentence(
            title="Ed Wood",
            sentence_id=0,
            text="Ed Wood was an American filmmaker.",
        ),
        Sentence(
            title="Other",
            sentence_id=0,
            text="This is an unrelated sentence.",
        ),
    ]

    retriever = ContractEvidenceRetriever()

    # --------------------------------------------------
    # Test 1: Existing Top-k retrieval
    # --------------------------------------------------
    results = retriever.retrieve(
        query="Scott Derrickson nationality",
        context=context,
        top_k=2,
    )

    assert len(results) == 2
    assert all(isinstance(item, Evidence) for item in results)

    for item in results:
        print(
            item.evidence_id,
            f"score={item.score:.4f}",
            f"rank={item.rank}",
        )

    print("Top-k retrieval test passed.")

    # --------------------------------------------------
    # Test 2: Top-10 support
    # --------------------------------------------------
    # Context has only 3 sentences, so all 3 are returned.
    top10_results = retriever.retrieve(
        query="Scott Derrickson nationality",
        context=context,
        top_k=10,
    )

    assert len(top10_results) == 3
    assert all(isinstance(item, Evidence) for item in top10_results)

    print("Top-10 retrieval test passed.")

    # --------------------------------------------------
    # Test 3: Full-context conversion
    # --------------------------------------------------
    full_results = retriever.full_context(context)

    assert len(full_results) == len(context)
    assert all(isinstance(item, Evidence) for item in full_results)

    # Original sentence order must be preserved.
    assert [
        (item.doc_title, item.sentence_id)
        for item in full_results
    ] == [
        (sentence.title, sentence.sentence_id)
        for sentence in context
    ]

    assert [item.rank for item in full_results] == [1, 2, 3]

    assert all(
        item.retrieval_query == "FULL_CONTEXT"
        for item in full_results
    )

    print("Full-context conversion test passed.")

    # --------------------------------------------------
    # Test 4: Empty context
    # --------------------------------------------------
    empty_results = retriever.full_context([])

    assert empty_results == []

    print("Empty-context test passed.")
    print("\nAll Contract Adapter tests passed.")


if __name__ == "__main__":
    main()