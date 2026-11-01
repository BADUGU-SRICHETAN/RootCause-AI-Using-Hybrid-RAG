from langchain_core.documents import Document

from src.preprocessing.text_cleaner import TextCleaner


def test_text_cleaning():

    cleaner = TextCleaner()

    sample_text = """
    Payment Service Error

    HTTP 500 - Database connection timeout


    Deployment ID: DEP-102
    """

    cleaned = cleaner.clean_text(sample_text)

    assert "HTTP 500" in cleaned
    assert "Database connection timeout" in cleaned
    assert "DEP-102" in cleaned

    print("Text cleaning test passed!")


def test_document_cleaning():

    cleaner = TextCleaner()

    documents = [
        Document(
            page_content="   Database connection failed   ",
            metadata={"source": "incident_001"}
        ),
        Document(
            page_content="   ",
            metadata={"source": "empty_record"}
        )
    ]

    cleaned_documents = cleaner.clean_documents(documents)

    assert len(cleaned_documents) == 1
    assert cleaned_documents[0].page_content == "Database connection failed"
    assert cleaned_documents[0].metadata["source"] == "incident_001"

    print("Document preprocessing test passed!")


if __name__ == "__main__":
    test_text_cleaning()
    test_document_cleaning()