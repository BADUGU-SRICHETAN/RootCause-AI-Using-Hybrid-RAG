from src.ingestion.document_loader import DocumentLoader


def test_document_ingestion():

    loader = DocumentLoader("data/raw")

    documents = loader.load_documents()

    assert len(documents) > 0

    for document in documents:

        assert document.page_content.strip()

        assert "source" in document.metadata

    print("\nIngestion test passed!")


if __name__ == "__main__":

    test_document_ingestion()