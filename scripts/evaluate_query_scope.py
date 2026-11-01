import json
from pathlib import Path

from src.analysis.query_scope_checker import QueryScopeChecker


EVALUATION_FILE = "data/evaluation/retrieval_test_cases.json"


def main():

    print("\n========== QUERY SCOPE EVALUATION ==========")

    with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
        cases = json.load(file)

    checker = QueryScopeChecker()

    tp = fp = tn = fn = 0

    results = []

    for case in cases:

        result = checker.check(case["query"])

        predicted_answerable = result["in_scope"]
        actual_answerable = case["answerable"]

        if predicted_answerable and actual_answerable:
            tp += 1

        elif predicted_answerable and not actual_answerable:
            fp += 1

        elif not predicted_answerable and not actual_answerable:
            tn += 1

        else:
            fn += 1

        results.append({
            "case_id": case["case_id"],
            "query": case["query"],
            "actual_answerable": actual_answerable,
            "predicted_answerable": predicted_answerable,
            "category": result["category"],
            "matched_pattern": result["matched_pattern"]
        })

    total = len(cases)

    accuracy = (tp + tn) / total if total else 0

    recall = tp / (tp + fn) if tp + fn else 0

    specificity = (
        tn / (tn + fp)
        if tn + fp else 0
    )

    print(f"\nTotal queries: {total}")
    print(f"True positives: {tp}")
    print(f"False positives: {fp}")
    print(f"True negatives: {tn}")
    print(f"False negatives: {fn}")

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")

    print("\n========== MISCLASSIFIED QUERIES ==========")

    for result in results:

        if (
            result["actual_answerable"]
            != result["predicted_answerable"]
        ):

            print("\nCase:", result["case_id"])
            print("Query:", result["query"])
            print("Actual:", result["actual_answerable"])
            print("Predicted:", result["predicted_answerable"])
            print("Category:", result["category"])
            print("Matched pattern:", result["matched_pattern"])

    output = Path("reports/query_scope_evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as file:
        json.dump({
            "metrics": {
                "total": total,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "accuracy": accuracy,
                "recall": recall,
                "specificity": specificity
            },
            "results": results
        }, file, indent=4)

    print("\nEvaluation completed.")
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()