import json
from pathlib import Path

from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.text_cleaner import TextCleaner

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


DATA_DIRECTORY = "data/raw"
EVALUATION_FILE = "data/evaluation/retrieval_test_cases.json"

PERSIST_DIRECTORY = "vectorstore/abstention_calibration"

TOP_K = 3
CANDIDATE_K = 10


def load_cases():

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(records, signal, threshold):

    tp = fp = tn = fn = 0

    for record in records:

        score = record[signal]

        predicted_answerable = (
            score is not None and score >= threshold
        )

        actual_answerable = record["answerable"]

        if predicted_answerable and actual_answerable:
            tp += 1

        elif predicted_answerable and not actual_answerable:
            fp += 1

        elif not predicted_answerable and not actual_answerable:
            tn += 1

        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    specificity = (
        tn / (tn + fp)
        if tn + fp else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4)
    }


def main():

    print("\n========== ABSTENTION CALIBRATION ==========")

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

    cases = load_cases()

    records = []

    print("\nCollecting query-level retrieval scores...")

    for case in cases:

        results = hybrid.retrieve(
            query=case["query"],
            top_k=TOP_K,
            candidate_k=CANDIDATE_K
        )

        metadata = (
            results[0].metadata
            if results else {}
        )

        ranks = metadata.get("retrieval_ranks", {})

        records.append({
            "case_id": case["case_id"],
            "category": case["category"],
            "query": case["query"],
            "answerable": case["answerable"],
            "vector_score": metadata.get("vector_score"),
            "bm25_score": metadata.get("bm25_score"),
            "rrf_score": metadata.get("rrf_score"),
            "retrieval_agreement": (
                "bm25" in ranks and "vector" in ranks
            )
        })

    # Test different thresholds.
    threshold_ranges = {
        "vector_score": [
            round(x / 100, 2)
            for x in range(10, 81, 5)
        ],
        "bm25_score": [
            float(x)
            for x in range(1, 16)
        ],
        "rrf_score": [
            round(x / 1000, 4)
            for x in range(10, 41, 2)
        ]
    }

    report = {
        "total_queries": len(records),
        "answerable": sum(r["answerable"] for r in records),
        "unanswerable": sum(not r["answerable"] for r in records),
        "threshold_results": {},
        "query_scores": records
    }

    for signal, thresholds in threshold_ranges.items():

        print(f"\n{'=' * 60}")
        print(f"Testing {signal}")
        print("=" * 60)

        signal_results = []

        for threshold in thresholds:

            metrics = calculate_metrics(
                records,
                signal,
                threshold
            )

            result = {
                "threshold": threshold,
                **metrics
            }

            signal_results.append(result)

            print(
                f"Threshold: {threshold:<8} "
                f"TP: {metrics['TP']:<3} "
                f"FP: {metrics['FP']:<3} "
                f"TN: {metrics['TN']:<3} "
                f"FN: {metrics['FN']:<3} "
                f"Recall: {metrics['recall']:.4f} "
                f"Specificity: {metrics['specificity']:.4f}"
            )

        report["threshold_results"][signal] = signal_results

    output_directory = Path("reports")
    output_directory.mkdir(parents=True, exist_ok=True)

    output_file = output_directory / "abstention_calibration.json"

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nCalibration completed.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()