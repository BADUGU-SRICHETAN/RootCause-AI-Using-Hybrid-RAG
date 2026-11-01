from langchain_core.documents import Document

from src.retrieval.bm25_retriever import BM25Retriever


def test_bm25_retrieval():

    documents = [
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

    retriever = BM25Retriever(documents)

    query = "Payment Service HTTP 500 database connection failure"

    results = retriever.retrieve(query, top_k=2)

    assert len(results) > 0

    assert results[0].metadata["incident_id"] == "INC001"

    assert "bm25_score" in results[0].metadata

    print("\nBM25 Retrieval Test Passed!")

    print("\nRetrieved Documents:")

    for result in results:
        print(
            result.metadata["incident_id"],
            result.metadata["bm25_score"]
        )


def test_empty_query():

    documents = [
        Document(page_content="Database connection failed.")
    ]

    retriever = BM25Retriever(documents)

    try:
        retriever.retrieve("")
        assert False, "Expected ValueError"
    except ValueError:
        print("\nEmpty query validation passed!")


if __name__ == "__main__":
    test_bm25_retrieval()
    test_empty_query()
