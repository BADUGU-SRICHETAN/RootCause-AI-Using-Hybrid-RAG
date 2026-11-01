import json
from pathlib import Path

from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.text_cleaner import TextCleaner

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


DATA_DIRECTORY = "data/raw"
EVALUATION_FILE = "data/evaluation/retrieval_test_cases.json"
CALIBRATION_FILE = "reports/abstention_calibration.json"

PERSIST_DIRECTORY = "vectorstore/abstention_error_inspection"

VECTOR_THRESHOLD = 0.35
TOP_K = 3
CANDIDATE_K = 10


def load_json(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    print("\n========== ABSTENTION ERROR INSPECTION ==========")

    # Load calibration and evaluation data
    calibration = load_json(CALIBRATION_FILE)
    evaluation_cases = load_json(EVALUATION_FILE)

    evaluation_map = {
        case["case_id"]: case
        for case in evaluation_cases
    }

    # Identify answerable queries rejected by the threshold
    rejected_cases = []

    for record in calibration["query_scores"]:

        if not record["answerable"]:
            continue

        score = record["vector_score"]

        if score is None or score < VECTOR_THRESHOLD:

            rejected_cases.append(record)

    print(f"\nVector threshold: {VECTOR_THRESHOLD}")
    print(f"Answerable queries rejected: {len(rejected_cases)}")

    if not rejected_cases:
        print("\nNo rejected answerable queries found.")
        return

    # Load and preprocess documents
    loader = DocumentLoader(DATA_DIRECTORY)
    documents = loader.load_documents()

    cleaner = TextCleaner()
    documents = cleaner.clean_documents(documents)

    # Initialize retrievers
    bm25 = BM25Retriever(documents)

    vector = VectorRetriever(
        documents=documents,
        persist_directory=PERSIST_DIRECTORY
    )

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector
    )

    inspection_results = []

    print("\n========== REJECTED QUERY DETAILS ==========")

    for record in rejected_cases:

        case_id = record["case_id"]
        case = evaluation_map[case_id]

        results = hybrid.retrieve(
            query=record["query"],
            top_k=TOP_K,
            candidate_k=CANDIDATE_K
        )

        print("\n" + "=" * 75)
        print(f"CASE ID: {case_id}")
        print(f"QUERY: {record['query']}")

        print(
            f"\nOriginal top-1 vector score: "
            f"{record['vector_score']}"
        )

        print(
            f"Expected relevant incidents: "
            f"{case['relevant_incident_ids']}"
        )

        retrieved_evidence = []

        print("\nRetrieved evidence:")

        for rank, document in enumerate(results, start=1):

            metadata = document.metadata

            incident_id = metadata.get("incident_id")

            evidence = {
                "rank": rank,
                "incident_id": incident_id,
                "vector_score": metadata.get("vector_score"),
                "bm25_score": metadata.get("bm25_score"),
                "rrf_score": metadata.get("rrf_score"),
                "retrieval_ranks": metadata.get("retrieval_ranks")
            }

            retrieved_evidence.append(evidence)

            print(f"\nRank: {rank}")
            print(f"Incident ID: {incident_id}")
            print(f"Vector score: {evidence['vector_score']}")
            print(f"BM25 score: {evidence['bm25_score']}")
            print(f"RRF score: {evidence['rrf_score']}")
            print(f"Retrieval ranks: {evidence['retrieval_ranks']}")

        expected_ids = set(case["relevant_incident_ids"])

        retrieved_ids = {
            item["incident_id"]
            for item in retrieved_evidence
            if item["incident_id"]
        }

        matched_ids = expected_ids.intersection(retrieved_ids)

        result = {
            "case_id": case_id,
            "query": record["query"],
            "original_vector_score": record["vector_score"],
            "expected_incident_ids": list(expected_ids),
            "retrieved_incident_ids": list(retrieved_ids),
            "matched_incident_ids": list(matched_ids),
            "retrieved_evidence": retrieved_evidence
        }

        inspection_results.append(result)

        print(f"\nRelevant incidents retrieved: {list(matched_ids)}")

        if matched_ids:
            print("Diagnosis: Relevant evidence was retrieved.")
        else:
            print("Diagnosis: Relevant evidence was not retrieved.")

    # Save inspection report
    output_directory = Path("reports")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = output_directory / "abstention_error_inspection.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(inspection_results, file, indent=4)

    print("\n========== INSPECTION COMPLETED ==========")
    print(f"Cases inspected: {len(inspection_results)}")
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()