from typing import Dict, Any

from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.text_cleaner import TextCleaner

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

from src.analysis.root_cause_analyzer import RootCauseAnalyzer
from src.analysis.abstention_checker import AbstentionChecker


class RootCausePipeline:

    def __init__(
        self,
        data_directory: str = "data/raw",
        persist_directory: str = "vectorstore/rootcause_chroma",
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small"
    ):

        print("\nInitializing RootCause AI Pipeline...")

        # 1. Load historical incident documents
        self.loader = DocumentLoader(data_directory)

        documents = self.loader.load_documents()

        if not documents:
            raise ValueError(
                "No documents were loaded. Check the data directory."
            )

        # 2. Clean documents
        self.cleaner = TextCleaner()

        cleaned_documents = self.cleaner.clean_documents(documents)

        if not cleaned_documents:
            raise ValueError(
                "No valid documents remain after preprocessing."
            )

        print(
            f"Documents after preprocessing: {len(cleaned_documents)}"
        )

        # 3. Initialize BM25 retrieval
        self.bm25_retriever = BM25Retriever(
            cleaned_documents
        )

        # 4. Initialize vector retrieval using OpenAI embeddings
        self.vector_retriever = VectorRetriever(
            documents=cleaned_documents,
            persist_directory=persist_directory,
            model_name=embedding_model,
            collection_name="rootcause_incidents_openai"
        )

        # 5. Initialize hybrid retrieval with RRF
        self.hybrid_retriever = HybridRetriever(
            bm25_retriever=self.bm25_retriever,
            vector_retriever=self.vector_retriever
        )

        # 6. Initialize abstention checker
        self.abstention_checker = AbstentionChecker(
            vector_score_threshold=0.35,
            score_top_k=3
        )

        # 7. Initialize LLM analyzer
        self.analyzer = RootCauseAnalyzer(
            model_name=model_name
        )

        print("\nRootCause AI Pipeline initialized successfully!")

        print(f"Embedding model: {embedding_model}")
        print(f"LLM model: {model_name}")

    def run(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 10
    ) -> Dict[str, Any]:

        if not query or not query.strip():
            raise ValueError("Incident query cannot be empty.")

        # 1. Check query scope before retrieval
        print("\nChecking query scope...")

        scope_result = self.abstention_checker.scope_checker.check(
            query
        )

        if not scope_result["in_scope"]:

            print("\nInvestigation rejected: Query is out of scope.")

            return {
                "query": query,
                "status": "abstained",
                "retrieved_count": 0,
                "retrieved_evidence": [],
                "abstention": {
                    "accepted": False,
                    "reason": "out_of_scope",
                    "category": scope_result["category"],
                    "message": scope_result["reason"],
                    "max_vector_score": None,
                    "threshold": 0.35
                },
                "analysis": (
                    "Investigation cannot proceed because the "
                    "question requires information outside the "
                    "historical incident database."
                )
            }

        # 2. Retrieve relevant historical incidents
        print("\nSearching historical incidents...")

        retrieved_documents = self.hybrid_retriever.retrieve(
            query=query,
            top_k=top_k,
            candidate_k=candidate_k
        )

        print(
            f"Retrieved {len(retrieved_documents)} relevant documents."
        )

        # 3. Check retrieved evidence
        print("\nEvaluating retrieved evidence...")

        abstention_result = self.abstention_checker.check(
            query=query,
            retrieved_documents=retrieved_documents
        )

        # 4. Prepare retrieved evidence
        retrieved_evidence = []

        for document in retrieved_documents:

            metadata = document.metadata

            retrieved_evidence.append({
                "incident_id": metadata.get("incident_id"),
                "source": metadata.get("source"),
                "content": document.page_content,
                "rrf_score": metadata.get("rrf_score"),
                "vector_score": metadata.get("vector_score"),
                "bm25_score": metadata.get("bm25_score"),
                "retrieval_ranks": metadata.get("retrieval_ranks")
            })

        # 5. Abstain if evidence is insufficient
        if not abstention_result["accepted"]:

            print("\nInvestigation rejected: Insufficient evidence.")

            return {
                "query": query,
                "status": "abstained",
                "retrieved_count": len(retrieved_documents),
                "retrieved_evidence": retrieved_evidence,
                "abstention": abstention_result,
                "analysis": abstention_result["message"]
            }

        # 6. Analyze evidence using GPT-4o-mini
        print("\nEvidence passed initial checks.")
        print("\nAnalyzing incident evidence...")

        analysis = self.analyzer.analyze(
            query=query,
            retrieved_documents=retrieved_documents
        )

        # 7. Prepare investigation report
        result = {
            "query": query,
            "status": "completed",
            "retrieved_count": len(retrieved_documents),
            "retrieved_evidence": retrieved_evidence,
            "abstention": abstention_result,
            "analysis": analysis
        }

        print("\nInvestigation completed successfully!")

        return result