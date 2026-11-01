import pytest

from langchain_core.documents import Document

from src.pipeline.root_cause_pipeline import RootCausePipeline
from src.analysis.abstention_checker import AbstentionChecker


class FakeHybridRetriever:

    def retrieve(self, query, top_k, candidate_k):

        return [
            Document(
                page_content=(
                    "Payment Service encountered HTTP 500 errors "
                    "because the database connection pool was exhausted."
                ),
                metadata={
                    "incident_id": "INC001",
                    "rrf_score": 0.032,
                    "vector_score": 0.42,
                    "bm25_score": 12.5,
                    "retrieval_ranks": {
                        "bm25": 1,
                        "vector": 1
                    }
                }
            )
        ]


class FakeAnalyzer:

    def analyze(self, query, retrieved_documents):

        assert query == "Why is Payment Service returning HTTP 500?"

        assert len(retrieved_documents) == 1

        return "Potential cause: Database connection pool exhaustion."


@pytest.fixture
def pipeline():

    instance = RootCausePipeline.__new__(RootCausePipeline)

    instance.hybrid_retriever = FakeHybridRetriever()

    instance.abstention_checker = AbstentionChecker(
        vector_score_threshold=0.35,
        score_top_k=3
    )

    instance.analyzer = FakeAnalyzer()

    return instance


def test_pipeline_execution(pipeline):

    result = pipeline.run(
        query="Why is Payment Service returning HTTP 500?",
        top_k=5,
        candidate_k=10
    )

    assert result["status"] == "completed"

    assert result["retrieved_count"] == 1

    assert result["retrieved_evidence"][0]["incident_id"] == "INC001"

    assert result["abstention"]["accepted"] is True

    assert result["abstention"]["max_vector_score"] == pytest.approx(0.42)

    assert "Database connection pool exhaustion" in result["analysis"]

    print("\nPipeline execution test passed!")


def test_empty_query(pipeline):

    with pytest.raises(ValueError):

        pipeline.run("")

    print("\nEmpty query validation passed!")


def test_out_of_scope_query_is_rejected(pipeline):

    result = pipeline.run(
        query="What will be the exact number of API requests tomorrow?"
    )

    assert result["status"] == "abstained"

    assert result["abstention"]["reason"] == "out_of_scope"

    assert result["retrieved_count"] == 0

    print("\nOut-of-scope query test passed!")