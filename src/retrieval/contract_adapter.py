
"""Adapter between B's MiniLM retriever and the shared Contract v0.1."""

from contracts.schemas import Evidence, Sentence
from src.retrieval.retriever import EvidenceRetriever


class ContractEvidenceRetriever:
    """Expose EvidenceRetriever through E's RetrievePort interface."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self._retriever = EvidenceRetriever(
            model_name=model_name
        )

    def retrieve(
        self,
        query: str,
        context: list[Sentence],
        top_k: int = 5,
    ) -> list[Evidence]:
        """Retrieve evidence using shared Contract dataclasses."""

        # Convert shared Sentence objects to B's internal format.
        internal_context = [
            {
                "title": sentence.title,
                "sentence_id": sentence.sentence_id,
                "text": sentence.text,
            }
            for sentence in context
        ]

        # Run the existing MiniLM retrieval algorithm.
        internal_results = self._retriever.retrieve(
            query=query,
            context=internal_context,
            top_k=top_k,
        )

        # Convert internal dictionaries to shared Evidence objects.
        return [
            Evidence(**result)
            for result in internal_results
        ]
