from typing import List
from hashlib import sha256

from langchain_core.documents import Document

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever


class HybridRetriever:

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: VectorRetriever,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        vector_weight: float = 1.0
    ):

        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        if bm25_weight < 0 or vector_weight < 0:
            raise ValueError("Retrieval weights cannot be negative.")

        if bm25_weight + vector_weight == 0:
            raise ValueError("At least one retrieval weight must be positive.")

        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever

        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    @staticmethod
    def _document_key(document: Document) -> str:

        content = document.page_content.strip()

        return sha256(
            content.encode("utf-8")
        ).hexdigest()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10
    ) -> List[Document]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be positive.")

        # Retrieve using BM25
        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=candidate_k
        )

        # Retrieve using semantic vector search
        vector_results = self.vector_retriever.retrieve(
            query=query,
            top_k=candidate_k
        )

        fused_documents = {}

        def add_results(
            results: List[Document],
            retriever_name: str,
            weight: float
        ):

            for rank, document in enumerate(results, start=1):

                key = self._document_key(document)

                if key not in fused_documents:

                    fused_documents[key] = {
                        "document": document,
                        "rrf_score": 0.0,
                        "retrieval_ranks": {}
                    }

                entry = fused_documents[key]

                entry["rrf_score"] += (
                    weight / (self.rrf_k + rank)
                )

                entry["retrieval_ranks"][retriever_name] = rank

                # Merge retriever-specific metadata
                entry["document"].metadata.update(
                    document.metadata
                )

        add_results(
            bm25_results,
            "bm25",
            self.bm25_weight
        )

        add_results(
            vector_results,
            "vector",
            self.vector_weight
        )

        # Sort by fused RRF score
        ranked_results = sorted(
            fused_documents.values(),
            key=lambda item: item["rrf_score"],
            reverse=True
        )

        final_results = []

        for entry in ranked_results[:top_k]:

            document = entry["document"]

            final_results.append(
                Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "rrf_score": entry["rrf_score"],
                        "retrieval_ranks": entry["retrieval_ranks"]
                    }
                )
            )

        return final_results
