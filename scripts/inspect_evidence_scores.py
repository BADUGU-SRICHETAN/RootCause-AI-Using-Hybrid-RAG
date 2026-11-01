import json
from pathlib import Path
from statistics import median

from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.text_cleaner import TextCleaner

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


DATA_DIRECTORY = "data/raw"
EVALUATION_FILE = "data/evaluation/retrieval_test_cases.json"

PERSIST_DIRECTORY = "vectorstore/evidence_calibration"

TOP_K = 3
CANDIDATE_K = 10


def load_evaluation_cases():

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def print_statistics(title, values):

    print(f"\n{title}")

    if not values:
        print("No values available.")
        return

    print(f"Count  : {len(values)}")
    print(f"Minimum: {min(values):.4f}")
    print(f"Median : {median(values):.4f}")
    print(f"Maximum: {max(values):.4f}")


def main():

    print("\n========== EVIDENCE SCORE DIAGNOSTICS ==========")

    # 1. Load incident records
    loader = DocumentLoader(DATA_DIRECTORY)
    documents = loader.load_documents()

    cleaner = TextCleaner()
    documents = cleaner.clean_documents(documents)

    print(f"\nLoaded incidents: {len(documents)}")

    # 2. Initialize retrievers
    bm25 = BM25Retriever(documents)

    vector = VectorRetriever(
        documents=documents,
        persist_directory=PERSIST_DIRECTORY
    )

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector
    )

    # 3. Load evaluation dataset
    cases = load_evaluation_cases()

    answerable_cases = [
        case for case in cases
        if case["answerable"] is True
    ]

    unanswerable_cases = [
        case for case in cases
        if case["answerable"] is False
    ]

    print(f"Answerable queries: {len(answerable_cases)}")
    print(f"Unanswerable queries: {len(unanswerable_cases)}")

    groups = {
        "ANSWERABLE": answerable_cases,
        "UNANSWERABLE": unanswerable_cases
    }

    all_results = {}

    # 4. Evaluate score distributions
    for group_name, group_cases in groups.items():

        print("\n" + "=" * 60)
        print(group_name)
        print("=" * 60)

        vector_scores = []
        bm25_scores = []
        rrf_scores = []

        top1_hits = 0
        top3_hits = 0
        agreement_count = 0

        for case in group_cases:

            query = case["query"]

            relevant_ids = set(
                case["relevant_incident_ids"]
            )

            results = hybrid.retrieve(
                query=query,
                top_k=TOP_K,
                candidate_k=CANDIDATE_K
            )

            retrieved_ids = [
                document.metadata.get("incident_id")
                for document in results
            ]

            if relevant_ids:

                if retrieved_ids:
                    if retrieved_ids[0] in relevant_ids:
                        top1_hits += 1

                if relevant_ids.intersection(retrieved_ids):
                    top3_hits += 1

            if results:

                top_document = results[0]
                metadata = top_document.metadata

                if metadata.get("vector_score") is not None:
                    vector_scores.append(
                        float(metadata["vector_score"])
                    )

                if metadata.get("bm25_score") is not None:
                    bm25_scores.append(
                        float(metadata["bm25_score"])
                    )

                if metadata.get("rrf_score") is not None:
                    rrf_scores.append(
                        float(metadata["rrf_score"])
                    )

                ranks = metadata.get("retrieval_ranks", {})

                if "bm25" in ranks and "vector" in ranks:
                    agreement_count += 1

        print(f"\nQueries evaluated: {len(group_cases)}")


        print(f"Top-1 relevant matches: {top1_hits}")
        print(f"Top-3 relevant matches: {top3_hits}")
        print(f"Top-1 BM25 + Vector agreement: {agreement_count}")

        print_statistics("Top-1 Vector Score", vector_scores)
        print_statistics("Top-1 BM25 Score", bm25_scores)
        print_statistics("Top-1 RRF Score", rrf_scores)

        all_results[group_name] = {
            "query_count": len(group_cases),
            "top1_hits": top1_hits,
            "top3_hits": top3_hits,
            "retrieval_agreement": agreement_count,
            "vector_scores": vector_scores,
            "bm25_scores": bm25_scores,
            "rrf_scores": rrf_scores
        }

    # 5. Save diagnostic results
    output_directory = Path("reports")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = output_directory / "evidence_score_diagnostics.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=4)

    print("\nDiagnostic results saved to:")
    print(output_file)

    print("\nEvidence score diagnostics completed!")


if __name__ == "__main__":
    main()