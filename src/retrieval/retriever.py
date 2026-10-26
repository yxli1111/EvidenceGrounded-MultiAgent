
from sentence_transformers import SentenceTransformer
import numpy as np


class EvidenceRetriever:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)

    def retrieve(
        self,
        query: str,
        context: list[dict],
        top_k: int = 5
    ) -> list[dict]:

        if not query.strip():
            raise ValueError("Retrieval query cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        if not context:
            return []

        texts = [sentence["text"] for sentence in context]

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        context_embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        similarities = (
            query_embedding @ context_embeddings.T
        )[0]

        ranked_indices = np.argsort(-similarities)[:top_k]

        results = []

        for rank, idx in enumerate(ranked_indices, start=1):
            sentence = context[int(idx)]

            results.append({
                "doc_title": sentence["title"],
                "sentence_id": sentence["sentence_id"],
                "text": sentence["text"],
                "score": float(similarities[idx]),
                "retrieval_query": query,
                "rank": rank
            })

        return results


    def retrieve_multi_query(
        self,
        queries: list[str],
        context: list[dict],
        total_k: int = 5
    ) -> list[dict]:
        """Retrieve evidence from multiple queries under a shared budget."""

        if total_k < 1:
            raise ValueError("total_k must be positive")

        if not queries or not context:
            return []

        queries = [q for q in queries if q.strip()]

        if not queries:
            raise ValueError("All retrieval queries are empty")

        # Retrieve up to total_k candidates for each query.
        candidate_lists = [
            self.retrieve(
                query=query,
                context=context,
                top_k=total_k
            )
            for query in queries
        ]

        selected = []
        seen = set()

        # Round-robin across query rankings.
        for rank_idx in range(total_k):
            for candidates in candidate_lists:
                if rank_idx >= len(candidates):
                    continue

                item = candidates[rank_idx]
                key = (
                    item["doc_title"],
                    item["sentence_id"]
                )

                if key in seen:
                    continue

                seen.add(key)
                selected.append(item.copy())

                if len(selected) >= total_k:
                    break

            if len(selected) >= total_k:
                break

        # Rank now means selection order, not global cosine ranking.
        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank

        return selected
