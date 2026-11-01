import json
from pathlib import Path
from collections import defaultdict

from src.ingestion.document_loader import DocumentLoader
from src.preprocessing.text_cleaner import TextCleaner

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever

from src.evaluation.retriever_evaluator import RetrieverEvaluator


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_DIRECTORY = "data/raw"

EVALUATION_FILE = "data/evaluation/retrieval_test_cases.json"

PERSIST_DIRECTORY = "vectorstore/evaluation"

OUTPUT_FILE = "reports/retrieval_comparison.json"

EVALUATION_K_VALUES = [1, 2, 3]

UNANSWERABLE_TOP_K = 3


# --------------------------------------------------
# LOAD EVALUATION DATASET
# --------------------------------------------------

def load_evaluation_cases(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# VALIDATE EVALUATION DATASET
# --------------------------------------------------

def validate_evaluation_cases(cases, documents):

    if not cases:
        raise ValueError("Evaluation dataset is empty.")

    available_ids = {
        str(document.metadata["incident_id"])
        for document in documents
        if document.metadata.get("incident_id") is not None
    }

    case_ids = set()

    for case in cases:

        case_id = case.get("case_id")

        if not case_id:
            raise ValueError("Evaluation case is missing case_id.")

        if case_id in case_ids:
            raise ValueError(f"Duplicate case ID: {case_id}")

        case_ids.add(case_id)

        if not case.get("query", "").strip():
            raise ValueError(f"Empty query in {case_id}.")

        if case.get("answerable") is True:

            relevant_ids = case.get("relevant_incident_ids", [])

            if not relevant_ids:
                raise ValueError(
                    f"Answerable case {case_id} has no relevant IDs."
                )

            unknown_ids = set(relevant_ids) - available_ids

            if unknown_ids:
                raise ValueError(
                    f"Unknown incident IDs in {case_id}: {unknown_ids}"
                )

        elif case.get("answerable") is False:

            if case.get("relevant_incident_ids"):
                raise ValueError(
                    f"Unanswerable case {case_id} contains relevant IDs."
                )

        else:
            raise ValueError(
                f"Invalid answerable value in {case_id}."
            )


# --------------------------------------------------
# EVALUATE ONE RETRIEVER
# --------------------------------------------------

def evaluate_retriever(
    retriever_name,
    retriever,
    answerable_cases,
    unanswerable_cases
):

    print("\n" + "=" * 75)
    print(f"EVALUATING: {retriever_name}")
    print("=" * 75)

    evaluator = RetrieverEvaluator(retriever)

    retriever_results = {
        "overall": {},
        "category_results": {},
        "unanswerable_diagnostics": {}
    }

    # --------------------------------------------------
    # 1. OVERALL ANSWERABLE EVALUATION
    # --------------------------------------------------

    print("\nOVERALL ANSWERABLE EVALUATION")

    for k in EVALUATION_K_VALUES:

        print(f"\nEvaluating Top-{k}...")

        results = evaluator.evaluate(
            evaluation_cases=answerable_cases,
            k=k
        )

        retriever_results["overall"][f"top_{k}"] = results

        print(f"Precision@{k}: {results['precision_at_k']:.4f}")
        print(f"Recall@{k}: {results['recall_at_k']:.4f}")
        print(f"MRR: {results['reciprocal_rank']:.4f}")
        print(f"Hit Rate@{k}: {results['hit_rate_at_k']:.4f}")
        print(f"nDCG@{k}: {results['ndcg_at_k']:.4f}")

    # --------------------------------------------------
    # 2. CATEGORY-WISE EVALUATION
    # --------------------------------------------------

    print("\nCATEGORY-WISE EVALUATION")

    categories = defaultdict(list)

    for case in answerable_cases:

        category = case.get("category", "uncategorized")

        categories[category].append(case)

    for category, category_cases in categories.items():

        print("\n" + "-" * 60)
        print(f"Category: {category}")
        print(f"Queries: {len(category_cases)}")
        print("-" * 60)

        category_results = {}

        for k in EVALUATION_K_VALUES:

            results = evaluator.evaluate(
                evaluation_cases=category_cases,
                k=k
            )

            category_results[f"top_{k}"] = results

            print(
                f"Top-{k} | "
                f"Precision: {results['precision_at_k']:.4f} | "
                f"Recall: {results['recall_at_k']:.4f} | "
                f"MRR: {results['reciprocal_rank']:.4f} | "
                f"Hit Rate: {results['hit_rate_at_k']:.4f} | "
                f"nDCG: {results['ndcg_at_k']:.4f}"
            )

        retriever_results["category_results"][category] = {
            "query_count": len(category_cases),
            "results": category_results
        }

    # --------------------------------------------------
    # 3. UNANSWERABLE QUERY DIAGNOSTICS
    # --------------------------------------------------

    print("\nUNANSWERABLE QUERY DIAGNOSTICS")

    unanswerable_results = []

    for case in unanswerable_cases:

        query = case["query"]

        if isinstance(retriever, HybridRetriever):

            retrieved_documents = retriever.retrieve(
                query=query,
                top_k=UNANSWERABLE_TOP_K,
                candidate_k=UNANSWERABLE_TOP_K * 2
            )

        else:

            retrieved_documents = retriever.retrieve(
                query=query,
                top_k=UNANSWERABLE_TOP_K
            )

        retrieved_ids = [
            str(document.metadata["incident_id"])
            for document in retrieved_documents
            if document.metadata.get("incident_id") is not None
        ]

        diagnostic = {
            "case_id": case["case_id"],
            "query": query,
            "retrieved_count": len(retrieved_documents),
            "retrieved_incident_ids": retrieved_ids
        }

        unanswerable_results.append(diagnostic)

        print(f"\n{case['case_id']}: {query}")
        print(f"Retrieved incidents: {retrieved_ids}")

    retriever_results["unanswerable_diagnostics"] = {

        "total_queries": len(unanswerable_cases),

        "top_k": UNANSWERABLE_TOP_K,

        "retrieval_results": unanswerable_results

    }

    return retriever_results


# --------------------------------------------------
# DISPLAY PERFORMANCE COMPARISON
# --------------------------------------------------

def display_comparison(all_results):

    print("\n" + "=" * 100)
    print("OVERALL RETRIEVAL PERFORMANCE COMPARISON")
    print("=" * 100)

    for k in EVALUATION_K_VALUES:

        print(f"\nTOP-{k} PERFORMANCE")

        print("-" * 100)

        print(
            f"{'Retriever':<15}"
            f"{'Precision':<13}"
            f"{'Recall':<13}"
            f"{'MRR':<13}"
            f"{'Hit Rate':<13}"
            f"{'nDCG':<13}"
        )

        print("-" * 100)

        for retriever_name, retriever_results in all_results.items():

            metrics = retriever_results["overall"][f"top_{k}"]

            print(
                f"{retriever_name:<15}"
                f"{metrics['precision_at_k']:<13.4f}"
                f"{metrics['recall_at_k']:<13.4f}"
                f"{metrics['reciprocal_rank']:<13.4f}"
                f"{metrics['hit_rate_at_k']:<13.4f}"
                f"{metrics['ndcg_at_k']:<13.4f}"
            )

        print("-" * 100)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\n========== ROOTCAUSE AI RETRIEVAL BENCHMARK ==========")

    # --------------------------------------------------
    # 1. LOAD INCIDENT RECORDS
    # --------------------------------------------------

    print("\nLoading historical incident records...")

    loader = DocumentLoader(DATA_DIRECTORY)

    documents = loader.load_documents()

    if not documents:
        raise ValueError("No historical incidents found.")

    # --------------------------------------------------
    # 2. PREPROCESS DOCUMENTS
    # --------------------------------------------------

    print("\nPreprocessing incident records...")

    cleaner = TextCleaner()

    cleaned_documents = cleaner.clean_documents(documents)

    print(f"Total incidents: {len(cleaned_documents)}")

    # --------------------------------------------------
    # 3. LOAD EVALUATION DATASET
    # --------------------------------------------------

    print("\nLoading evaluation dataset...")

    evaluation_cases = load_evaluation_cases(EVALUATION_FILE)

    validate_evaluation_cases(
        evaluation_cases,
        cleaned_documents
    )

    answerable_cases = [
        case for case in evaluation_cases
        if case["answerable"] is True
    ]

    unanswerable_cases = [
        case for case in evaluation_cases
        if case["answerable"] is False
    ]

    print(f"Total queries: {len(evaluation_cases)}")
    print(f"Answerable queries: {len(answerable_cases)}")
    print(f"Unanswerable queries: {len(unanswerable_cases)}")

    # --------------------------------------------------
    # 4. INITIALIZE RETRIEVERS
    # --------------------------------------------------

    print("\nInitializing BM25 retriever...")

    bm25_retriever = BM25Retriever(cleaned_documents)

    print("\nInitializing Vector retriever...")

    vector_retriever = VectorRetriever(
        documents=cleaned_documents,
        persist_directory=PERSIST_DIRECTORY,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("\nInitializing Hybrid RAG retriever...")

    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_retriever=vector_retriever
    )

    # --------------------------------------------------
    # 5. REGISTER RETRIEVERS
    # --------------------------------------------------

    retrievers = {
        "BM25": bm25_retriever,
        "Vector": vector_retriever,
        "Hybrid_RAG": hybrid_retriever
    }

    # --------------------------------------------------
    # 6. EVALUATE RETRIEVERS
    # --------------------------------------------------

    all_results = {}

    for retriever_name, retriever in retrievers.items():

        all_results[retriever_name] = evaluate_retriever(
            retriever_name=retriever_name,
            retriever=retriever,
            answerable_cases=answerable_cases,
            unanswerable_cases=unanswerable_cases
        )

    # --------------------------------------------------
    # 7. DISPLAY COMPARISON
    # --------------------------------------------------

    display_comparison(all_results)

    # --------------------------------------------------
    # 8. SAVE RESULTS
    # --------------------------------------------------

    output_directory = Path("reports")

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = Path(OUTPUT_FILE)

    category_counts = {}

    for case in evaluation_cases:

        category = case.get("category", "uncategorized")

        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    report = {

        "project": "RootCause AI",

        "dataset_type": "synthetic",

        "total_incidents": len(cleaned_documents),

        "total_evaluation_queries": len(evaluation_cases),

        "answerable_queries": len(answerable_cases),

        "unanswerable_queries": len(unanswerable_cases),

        "category_distribution": category_counts,

        "retrievers": list(retrievers.keys()),

        "evaluation_k_values": EVALUATION_K_VALUES,

        "metrics": [
            "precision_at_k",
            "recall_at_k",
            "reciprocal_rank",
            "hit_rate_at_k",
            "ndcg_at_k"
        ],

        "unanswerable_evaluation_note": (
            "Retrieval diagnostics only. These results do not measure "
            "whether the LLM analyzer correctly abstains."
        ),

        "results": all_results

    }

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------
    # 9. FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("EVALUATION COMPLETED")
    print("=" * 65)

    print(f"Total incidents: {len(cleaned_documents)}")
    print(f"Total evaluation queries: {len(evaluation_cases)}")
    print(f"Answerable queries: {len(answerable_cases)}")
    print(f"Unanswerable queries: {len(unanswerable_cases)}")
    print(f"Total retrievers evaluated: {len(retrievers)}")
    print("Retrieval metrics evaluated: 5")

    print(f"\nResults saved to: {output_file}")

    print("\nRootCause AI retrieval benchmark completed!")


if __name__ == "__main__":

    main()