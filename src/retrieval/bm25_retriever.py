import re
from typing import List

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class BM25Retriever:

    def __init__(self, documents: List[Document]):

        if not documents:
            raise ValueError("Documents cannot be empty.")

        self.documents = documents

        self.tokenized_documents = [
            self.tokenize(doc.page_content)
            for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

    @staticmethod
    def tokenize(text: str) -> List[str]:

        # Preserve technical identifiers and error codes.
        return re.findall(
            r"[a-zA-Z0-9_./:-]+",
            text.lower()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Document]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True
        )

        results = []

        for index in ranked_indices:

            score = float(scores[index])

            if score <= 0:
                continue

            original_document = self.documents[index]

            result = Document(
                page_content=original_document.page_content,
                metadata={
                    **original_document.metadata,
                    "bm25_score": score
                }
            )

            results.append(result)

            if len(results) >= top_k:
                break

        return results
