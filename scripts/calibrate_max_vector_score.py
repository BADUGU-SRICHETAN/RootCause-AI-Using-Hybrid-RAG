import json
from pathlib import Path

from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.text_cleaner import TextCleaner

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


DATA_DIRECTORY = "data/raw"
EVALUATION_FILE = "data/evaluation/retrieval_test_cases.json"

PERSIST_DIRECTORY = "vectorstore/max_vector_calibration"

TOP_K = 3
CANDIDATE_K = 10


def main():

    print("\n========== MAX VECTOR SCORE CALIBRATION ==========")

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        cases = json.load(file)

    loader = DocumentLoader(DATA_DIRECTORY)
    documents = loader.load_documents()

    cleaner = TextCleaner()
    documents = cleaner.clean_documents(documents)

    bm25 = BM25Retriever(documents)

    vector = VectorRetriever(
        documents=documents,
        persist_directory=PERSIST_DIRECTORY
    )

    hybrid = HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector
    )

    records = []

    print("\nCollecting maximum vector scores...")

    for case in cases:

        results = hybrid.retrieve(
            query=case["query"],
            top_k=TOP_K,
            candidate_k=CANDIDATE_K
        )

        vector_scores = [
            document.metadata["vector_score"]
            for document in results
            if document.metadata.get("vector_score") is not None
        ]

        max_vector_score = (
            max(vector_scores)
            if vector_scores else None
        )

        retrieved_ids = [
            document.metadata.get("incident_id")
            for document in results
        ]

        relevant_ids = set(case["relevant_incident_ids"])

        matched_ids = (
            set(retrieved_ids) & relevant_ids
        )

        records.append({
            "case_id": case["case_id"],
            "category": case["category"],
            "query": case["query"],
            "answerable": case["answerable"],
            "max_vector_score": max_vector_score,
            "retrieved_incident_ids": retrieved_ids,
            "relevant_incident_ids": list(relevant_ids),
            "matched_incident_ids": list(matched_ids)
        })

    thresholds = [
        round(x / 100, 2)
        for x in range(10, 81, 5)
    ]

    report = {
        "total_queries": len(records),
        "top_k": TOP_K,
        "threshold_results": [],
        "query_scores": records
    }

    print("\n========== THRESHOLD RESULTS ==========")

    for threshold in thresholds:

        tp = fp = tn = fn = 0

        for record in records:

            score = record["max_vector_score"]

            predicted = (
                score is not None and score >= threshold
            )

            actual = record["answerable"]

            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and not actual:
                tn += 1
            else:
                fn += 1

        recall = tp / (tp + fn) if tp + fn else 0
        specificity = tn / (tn + fp) if tn + fp else 0

        result = {
            "threshold": threshold,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "recall": round(recall, 4),
            "specificity": round(specificity, 4)
        }

        report["threshold_results"].append(result)

        print(
            f"Threshold: {threshold:.2f} | "
            f"TP: {tp} | FP: {fp} | "
            f"TN: {tn} | FN: {fn} | "
            f"Recall: {recall:.4f} | "
            f"Specificity: {specificity:.4f}"
        )

    output = Path("reports/max_vector_calibration.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nCalibration completed.")
    print(f"Saved report: {output}")


if __name__ == "__main__":
    main()