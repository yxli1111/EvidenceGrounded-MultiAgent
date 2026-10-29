
import numpy as np
from sentence_transformers import SentenceTransformer


class EvidenceRetriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def retrieve(
        self,
        query: str,
        context: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """Retrieve top-k sentences by cosine similarity."""

        if not query.strip():
            raise ValueError("Retrieval query cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be positive")

        if not context:
            return []

        texts = [sentence["text"] for sentence in context]

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        context_embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
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
                "rank": rank,
            })

        return results

    def retrieve_multi_query(
        self,
        queries: list[str],
        context: list[dict],
        total_k: int = 5,
    ) -> list[dict]:
        """Round-robin multi-query retrieval with shared evidence budget."""

        if total_k < 1:
            raise ValueError("total_k must be positive")

        if not queries or not context:
            return []

        queries = [q for q in queries if q.strip()]

        if not queries:
            raise ValueError("All retrieval queries are empty")

        candidate_lists = [
            self.retrieve(
                query=query,
                context=context,
                top_k=total_k,
            )
            for query in queries
        ]

        selected = []
        seen = set()

        for rank_idx in range(total_k):
            for candidates in candidate_lists:
                if rank_idx >= len(candidates):
                    continue

                item = candidates[rank_idx]

                key = (
                    item["doc_title"],
                    item["sentence_id"],
                )

                if key in seen:
                    continue

                seen.add(key)
                selected.append(item.copy())

                if len(selected) >= total_k:
                    break

            if len(selected) >= total_k:
                break

        for rank, item in enumerate(selected, start=1):
            item["rank"] = rank

        return selected

    def retrieve_with_neighbors(
        self,
        query: str,
        context: list[dict],
        total_k: int = 5,
        seed_k: int = 3,
        window: int = 1,
    ) -> list[dict]:
        """
        Retrieve semantic seeds, then expand to adjacent sentences.

        Final output contains at most total_k distinct sentences.
        """

        if not query.strip():
            raise ValueError("Retrieval query cannot be empty")

        if total_k < 1:
            raise ValueError("total_k must be positive")

        if seed_k < 1 or seed_k > total_k:
            raise ValueError(
                "seed_k must be between 1 and total_k"
            )

        if window < 0:
            raise ValueError("window must be non-negative")

        if not context:
            return []

        # Compute embeddings once for this retrieval call.
        texts = [sentence["text"] for sentence in context]

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        )

        context_embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
        )

        similarities = (
            query_embedding @ context_embeddings.T
        )[0]

        ranked_indices = np.argsort(-similarities)

        # Lookup by original HotpotQA sentence identifiers.
        sentence_lookup = {
            (s["title"], s["sentence_id"]): i
            for i, s in enumerate(context)
        }

        selected = []
        seen = set()

        def add_index(idx: int, source: str):
            sentence = context[idx]

            key = (
                sentence["title"],
                sentence["sentence_id"],
            )

            if key in seen or len(selected) >= total_k:
                return

            seen.add(key)

            selected.append({
                "doc_title": sentence["title"],
                "sentence_id": sentence["sentence_id"],
                "text": sentence["text"],
                "score": float(similarities[idx]),
                "retrieval_query": query,
                "rank": len(selected) + 1,
                "selection_source": source,
            })

        # Step 1: Select semantic seed sentences.
        seed_indices = [
            int(i) for i in ranked_indices[:seed_k]
        ]

        for idx in seed_indices:
            add_index(idx, "semantic")

        # Step 2: Add neighboring sentences.
        for distance in range(1, window + 1):
            for idx in seed_indices:
                seed = context[idx]

                for offset in (-distance, distance):
                    neighbor_key = (
                        seed["title"],
                        seed["sentence_id"] + offset,
                    )

                    neighbor_idx = sentence_lookup.get(
                        neighbor_key
                    )

                    if neighbor_idx is not None:
                        add_index(
                            neighbor_idx,
                            "neighbor",
                        )

        # Step 3: Fill remaining budget using semantic ranking.
        for idx in ranked_indices:
            if len(selected) >= total_k:
                break

            add_index(int(idx), "semantic_fallback")

        return selected
