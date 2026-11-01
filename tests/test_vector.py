import uuid

import pytest
from langchain_core.documents import Document

from src.retrieval.vector_retriever import VectorRetriever


@pytest.fixture
def sample_documents():

    return [
        Document(
            page_content=(
                "Payment Service experienced HTTP 500 errors "
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


def test_vector_retrieval(sample_documents, tmp_path):

    retriever = VectorRetriever(
        documents=sample_documents,
        persist_directory=str(tmp_path / "chroma"),
        collection_name=f"test_{uuid.uuid4().hex[:12]}"
    )

    query = (
        "Why is the payment application unable "
        "to establish a database connection?"
    )

    results = retriever.retrieve(query, top_k=2)

    assert len(results) == 2

    assert results[0].metadata["incident_id"] == "INC001"

    assert "vector_score" in results[0].metadata
    assert "vector_distance" in results[0].metadata

    print("\nVector Retrieval Test Passed!")

    print("\nRetrieved Documents:")

    for result in results:

        print(
            result.metadata["incident_id"],
            result.metadata["vector_score"]
        )


def test_empty_query(sample_documents, tmp_path):

    retriever = VectorRetriever(
        documents=sample_documents,
        persist_directory=str(tmp_path / "chroma"),
        collection_name=f"test_{uuid.uuid4().hex[:12]}"
    )

    with pytest.raises(ValueError):
        retriever.retrieve("")

    print("\nEmpty query validation passed!")
