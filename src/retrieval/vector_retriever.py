import os
from pathlib import Path
from typing import List

import chromadb
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


load_dotenv()


class VectorRetriever:

    def __init__(
        self,
        documents: List[Document],
        persist_directory: str = "vectorstore/rootcause_chroma",
        model_name: str = "text-embedding-3-small",
        collection_name: str = "rootcause_incidents_openai"
    ):

        if not documents:
            raise ValueError("Documents cannot be empty.")

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it to your .env file."
            )

        self.documents = documents

        # Initialize OpenAI embeddings instead of local Sentence Transformers.
        self.embeddings = OpenAIEmbeddings(
            model=model_name
        )

        # Create persistent ChromaDB client.
        Path(persist_directory).mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        # Use a new collection because OpenAI embeddings
        # have a different dimensionality from MiniLM embeddings.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Generate document embeddings.
        texts = [
            document.page_content
            for document in documents
        ]

        embeddings = self.embeddings.embed_documents(texts)

        # Prepare metadata.
        metadatas = [
            self._sanitize_metadata(document.metadata)
            for document in documents
        ]

        ids = [
            f"document_{index}"
            for index in range(len(documents))
        ]

        # Store documents and embeddings.
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    @staticmethod
    def _sanitize_metadata(metadata: dict) -> dict:

        clean_metadata = {}

        for key, value in metadata.items():

            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                clean_metadata[key] = value
            else:
                clean_metadata[key] = str(value)

        return clean_metadata or {"source": "unknown"}

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Document]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        collection_count = self.collection.count()

        if collection_count == 0:
            return []

        # Generate query embedding through OpenAI.
        query_embedding = self.embeddings.embed_query(query)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(
                top_k,
                collection_count
            ),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved_documents = []

        for text, metadata, distance in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0]
        ):

            similarity_score = 1.0 - float(distance)

            retrieved_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        **metadata,
                        "vector_distance": float(distance),
                        "vector_score": similarity_score
                    }
                )
            )

        return retrieved_documents