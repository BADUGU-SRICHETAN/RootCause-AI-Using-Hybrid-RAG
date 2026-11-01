import pytest

from langchain_core.documents import Document

from src.analysis.root_cause_analyzer import RootCauseAnalyzer


@pytest.fixture
def analyzer(monkeypatch):

    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-api-key"
    )

    return RootCauseAnalyzer()


def test_evidence_context(analyzer):

    documents = [
        Document(
            page_content=(
                "Payment Service returned HTTP 500 "
                "because the database connection pool was exhausted."
            ),
            metadata={"incident_id": "INC001"}
        )
    ]

    context = analyzer.build_evidence_context(documents)

    assert "INC001" in context
    assert "HTTP 500" in context
    assert "database connection pool" in context

    print("\nEvidence context test passed!")


def test_empty_query(analyzer):

    with pytest.raises(ValueError):
        analyzer.analyze(
            query="",
            retrieved_documents=[]
        )

    print("\nEmpty query validation passed!")


def test_no_retrieved_evidence(analyzer):

    response = analyzer.analyze(
        query="Why is the payment service returning HTTP 500?",
        retrieved_documents=[]
    )

    assert "Insufficient evidence" in response

    print("\nInsufficient evidence handling passed!")
