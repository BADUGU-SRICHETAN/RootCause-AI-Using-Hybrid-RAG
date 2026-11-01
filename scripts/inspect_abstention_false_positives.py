import json
from pathlib import Path


CALIBRATION_FILE = "reports/max_vector_calibration.json"

THRESHOLD = 0.35


def main():

    print("\n========== FALSE POSITIVE INSPECTION ==========")

    with open(CALIBRATION_FILE, "r", encoding="utf-8") as file:
        report = json.load(file)

    false_positives = []

    for record in report["query_scores"]:

        if record["answerable"]:
            continue

        score = record["max_vector_score"]

        if score is not None and score >= THRESHOLD:
            false_positives.append(record)

    print(f"\nThreshold: {THRESHOLD}")
    print(f"False positives found: {len(false_positives)}")

    for record in false_positives:

        print("\n" + "=" * 65)

        print(f"Case ID: {record['case_id']}")
        print(f"Category: {record['category']}")
        print(f"Query: {record['query']}")

        print(
            f"Maximum vector score: "
            f"{record['max_vector_score']:.4f}"
        )

        print(
            f"Retrieved incidents: "
            f"{record['retrieved_incident_ids']}"
        )

        print(
            f"Expected relevant incidents: "
            f"{record['relevant_incident_ids']}"
        )

        print(
            f"Matched incidents: "
            f"{record['matched_incident_ids']}"
        )

        print("Diagnosis: Unanswerable query incorrectly passed.")

    output = Path("reports/abstention_false_positives.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w", encoding="utf-8") as file:
        json.dump(false_positives, file, indent=4)

    print("\nInspection completed.")
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()