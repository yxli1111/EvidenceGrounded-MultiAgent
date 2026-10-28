
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

    print("Contract Adapter smoke test passed.")


if __name__ == "__main__":
    main()
