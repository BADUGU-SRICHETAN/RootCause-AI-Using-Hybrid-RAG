import uuid

import pytest

from langchain_core.documents import Document

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def sample_documents():

    return [
        Document(
            page_content=(
                "Payment Service encountered HTTP 500 errors "
                "because the database connection pool was exhausted."
            ),
            metadata={"incident_id": "INC001"}
        ),
        Document(
            page_content=(
                "User Service experienced a memory leak "
                "causing application crashes."
            ),
            metadata={"incident_id": "INC002"}
        ),
        Document(
            page_content=(
                "Authentication Service failed because "
                "a required deployment environment variable was missing."
            ),
            metadata={"incident_id": "INC003"}
        )
    ]


@pytest.fixture
def hybrid_retriever(sample_documents, tmp_path):

    bm25 = BM25Retriever(sample_documents)

    vector = VectorRetriever(
        documents=sample_documents,
        persist_directory=str(tmp_path / "chroma"),
        collection_name=f"hybrid_{uuid.uuid4().hex[:12]}"
    )

    return HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector,
        rrf_k=60
    )


def test_hybrid_retrieval(hybrid_retriever):

    query = (
        "Why is the payment application unable "
        "to establish a database connection?"
    )

    results = hybrid_retriever.retrieve(
        query=query,
        top_k=2,
        candidate_k=3
    )

    assert len(results) == 2

    assert results[0].metadata["incident_id"] == "INC001"

    assert "rrf_score" in results[0].metadata

    assert "bm25" in results[0].metadata["retrieval_ranks"]

    assert "vector" in results[0].metadata["retrieval_ranks"]

    print("\nHybrid Retrieval Test Passed!")

    print("\nFused Results:")

    for document in results:

        print(
            document.metadata["incident_id"],
            document.metadata["rrf_score"],
            document.metadata["retrieval_ranks"]
        )


def test_empty_query(hybrid_retriever):

    with pytest.raises(ValueError):
        hybrid_retriever.retrieve("")

    print("\nEmpty query validation passed!")


def test_invalid_top_k(hybrid_retriever):

    with pytest.raises(ValueError):
        hybrid_retriever.retrieve(
            query="Database connection failure",
            top_k=0
        )

    print("\nInvalid top_k validation passed!")
