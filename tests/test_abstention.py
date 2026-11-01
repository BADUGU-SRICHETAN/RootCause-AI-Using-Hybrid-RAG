import pytest

from langchain_core.documents import Document

from src.analysis.abstention_checker import AbstentionChecker


@pytest.fixture
def checker():
    return AbstentionChecker(
        vector_score_threshold=0.35,
        score_top_k=3
    )


def create_document(vector_score=None):

    metadata = {}

    if vector_score is not None:
        metadata["vector_score"] = vector_score

    return Document(
        page_content="Database connection pool exhaustion.",
        metadata=metadata
    )


def test_out_of_scope_query_is_rejected(checker):

    result = checker.check(
        query="What will be the exact number of API requests tomorrow?"
    )

    assert result["accepted"] is False
    assert result["reason"] == "out_of_scope"


def test_empty_retrieved_documents_are_rejected(checker):

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=[]
    )

    assert result["accepted"] is False
    assert result["reason"] == "no_evidence"


def test_weak_evidence_is_rejected(checker):

    documents = [
        create_document(0.20),
        create_document(0.25),
        create_document(0.30)
    ]

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=documents
    )

    assert result["accepted"] is False
    assert result["reason"] == "weak_evidence"


def test_sufficient_evidence_is_accepted(checker):

    documents = [
        create_document(0.20),
        create_document(0.42),
        create_document(0.30)
    ]

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=documents
    )

    assert result["accepted"] is True
    assert result["reason"] == "evidence_available"
    assert result["max_vector_score"] == pytest.approx(0.42)


def test_missing_vector_scores_are_ignored(checker):

    documents = [
        create_document(),
        create_document(0.40),
        create_document(0.25)
    ]

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=documents
    )

    assert result["accepted"] is True
    assert result["max_vector_score"] == pytest.approx(0.40)


def test_no_vector_scores_are_rejected(checker):

    documents = [
        create_document(),
        create_document(),
        create_document()
    ]

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=documents
    )

    assert result["accepted"] is False
    assert result["reason"] == "no_vector_score"


def test_only_top_three_documents_are_scored(checker):

    documents = [
        create_document(0.20),
        create_document(0.25),
        create_document(0.30),
        create_document(0.90)
    ]

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=documents
    )

    assert result["accepted"] is False
    assert result["max_vector_score"] == pytest.approx(0.30)


def test_empty_query_is_rejected(checker):

    with pytest.raises(ValueError):
        checker.check(
            query="",
            retrieved_documents=[]
        )


def test_threshold_boundary_is_accepted(checker):

    documents = [
        create_document(0.35)
    ]

    result = checker.check(
        query="Why is the Payment Service returning HTTP 500 errors?",
        retrieved_documents=documents
    )

    assert result["accepted"] is True