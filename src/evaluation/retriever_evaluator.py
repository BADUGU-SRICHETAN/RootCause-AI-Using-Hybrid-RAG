from typing import Any, Dict, List, Set
import math

from src.retrieval.hybrid_retriever import HybridRetriever


class RetrieverEvaluator:

    def __init__(self, retriever):
        self.retriever = retriever

    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:

        if k <= 0:
            raise ValueError("k must be positive.")

        relevant_count = sum(
            1 for incident_id in retrieved_ids[:k]
            if incident_id in relevant_ids
        )

        return relevant_count / k

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:

        if k <= 0:
            raise ValueError("k must be positive.")

        if not relevant_ids:
            raise ValueError("Relevant incident IDs cannot be empty.")

        relevant_count = sum(
            1 for incident_id in retrieved_ids[:k]
            if incident_id in relevant_ids
        )

        return relevant_count / len(relevant_ids)

    @staticmethod
    def reciprocal_rank(
        retrieved_ids: List[str],
        relevant_ids: Set[str]
    ) -> float:

        for rank, incident_id in enumerate(
            retrieved_ids,
            start=1
        ):
            if incident_id in relevant_ids:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def hit_rate_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:

        if k <= 0:
            raise ValueError("k must be positive.")

        if not relevant_ids:
            raise ValueError("Relevant incident IDs cannot be empty.")

        return float(
            any(
                incident_id in relevant_ids
                for incident_id in retrieved_ids[:k]
            )
        )

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: List[str],
        relevant_ids: Set[str],
        k: int
    ) -> float:

        if k <= 0:
            raise ValueError("k must be positive.")

        if not relevant_ids:
            raise ValueError("Relevant incident IDs cannot be empty.")

        # Calculate actual DCG using retrieved documents.
        dcg = 0.0

        for rank, incident_id in enumerate(
            retrieved_ids[:k],
            start=1
        ):

            relevance = int(incident_id in relevant_ids)

            dcg += relevance / math.log2(rank + 1)

        # Calculate ideal DCG using ALL relevant documents.
        # Do not calculate IDCG from retrieved documents.
        ideal_relevant_count = min(k, len(relevant_ids))

        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_relevant_count + 1)
        )

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _retrieve(
        self,
        query: str,
        k: int
    ):

        if isinstance(self.retriever, HybridRetriever):

            return self.retriever.retrieve(
                query=query,
                top_k=k,
                candidate_k=max(k * 2, k)
            )

        return self.retriever.retrieve(
            query=query,
            top_k=k
        )

    def evaluate(
        self,
        evaluation_cases: List[Dict[str, Any]],
        k: int = 5
    ) -> Dict[str, Any]:

        if not evaluation_cases:
            raise ValueError("Evaluation dataset cannot be empty.")

        if k <= 0:
            raise ValueError("k must be positive.")

        results = []

        for case in evaluation_cases:

            query = case["query"]

            relevant_ids = set(
                case["relevant_incident_ids"]
            )

            if not query.strip():
                raise ValueError("Evaluation query cannot be empty.")

            if not relevant_ids:
                raise ValueError("Relevant incident IDs cannot be empty.")

            retrieved_documents = self._retrieve(
                query=query,
                k=k
            )

            retrieved_ids = [
                str(document.metadata["incident_id"])
                for document in retrieved_documents
                if document.metadata.get("incident_id") is not None
            ]

            metrics = {

                "precision_at_k": self.precision_at_k(
                    retrieved_ids, relevant_ids, k
                ),

                "recall_at_k": self.recall_at_k(
                    retrieved_ids, relevant_ids, k
                ),

                "reciprocal_rank": self.reciprocal_rank(
                    retrieved_ids, relevant_ids
                ),

                "hit_rate_at_k": self.hit_rate_at_k(
                    retrieved_ids, relevant_ids, k
                ),

                "ndcg_at_k": self.ndcg_at_k(
                    retrieved_ids, relevant_ids, k
                )

            }

            results.append({

                "query": query,

                "retrieved_ids": retrieved_ids,

                "relevant_ids": sorted(relevant_ids),

                **metrics

            })

        total = len(results)

        metric_keys = [
            "precision_at_k",
            "recall_at_k",
            "reciprocal_rank",
            "hit_rate_at_k",
            "ndcg_at_k"
        ]

        summary = {
            metric: sum(
                item[metric] for item in results
            ) / total
            for metric in metric_keys
        }

        return {

            "total_queries": total,

            "k": k,

            **summary,

            "individual_results": results

        }