from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from src.analysis.query_scope_checker import QueryScopeChecker


class AbstentionChecker:

    def __init__(
        self,
        vector_score_threshold: float = 0.35,
        score_top_k: int = 3
    ):

        if not 0 <= vector_score_threshold <= 1:
            raise ValueError(
                "Vector score threshold must be between 0 and 1."
            )

        if score_top_k <= 0:
            raise ValueError(
                "score_top_k must be greater than zero."
            )

        self.vector_score_threshold = vector_score_threshold
        self.score_top_k = score_top_k

        self.scope_checker = QueryScopeChecker()

    def check(
        self,
        query: str,
        retrieved_documents: Optional[List[Document]] = None
    ) -> Dict[str, Any]:

        # 1. Validate query
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        # 2. Check query scope before evidence evaluation
        scope_result = self.scope_checker.check(query)

        if not scope_result["in_scope"]:

            return {
                "accepted": False,
                "reason": "out_of_scope",
                "category": scope_result["category"],
                "message": scope_result["reason"],
                "max_vector_score": None,
                "threshold": self.vector_score_threshold
            }

        # 3. Check whether historical evidence was retrieved
        if not retrieved_documents:

            return {
                "accepted": False,
                "reason": "no_evidence",
                "category": "insufficient_evidence",
                "message": (
                    "No relevant historical incident evidence "
                    "was retrieved."
                ),
                "max_vector_score": None,
                "threshold": self.vector_score_threshold
            }

        # 4. Consider only the top-k retrieved documents
        scoring_documents = retrieved_documents[:self.score_top_k]

        vector_scores = []

        for document in scoring_documents:

            score = document.metadata.get("vector_score")

            # Ignore missing scores rather than treating them as zero
            if score is None:
                continue

            vector_scores.append(float(score))

        # 5. Handle cases where no vector scores are available
        if not vector_scores:

            return {
                "accepted": False,
                "reason": "no_vector_score",
                "category": "insufficient_evidence",
                "message": (
                    "No vector similarity scores were available "
                    "to evaluate the retrieved evidence."
                ),
                "max_vector_score": None,
                "threshold": self.vector_score_threshold
            }

        # 6. Calculate maximum vector similarity score
        max_vector_score = max(vector_scores)

        # 7. Apply calibrated threshold
        if max_vector_score < self.vector_score_threshold:

            return {
                "accepted": False,
                "reason": "weak_evidence",
                "category": "insufficient_evidence",
                "message": (
                    "Retrieved evidence did not meet the "
                    "minimum vector similarity threshold."
                ),
                "max_vector_score": max_vector_score,
                "threshold": self.vector_score_threshold
            }

        # 8. Accept query for further LLM analysis
        return {
            "accepted": True,
            "reason": "evidence_available",
            "category": "historical_incident_investigation",
            "message": (
                "The query passed scope and retrieval-score "
                "checks. Proceed with evidence-based analysis."
            ),
            "max_vector_score": max_vector_score,
            "threshold": self.vector_score_threshold
        }