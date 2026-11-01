import json

CALIBRATION_FILE = "reports/max_vector_calibration.json"

THRESHOLD = 0.35


def main():

    print("\n========== UNANSWERABLE QUERY ANALYSIS ==========")

    with open(CALIBRATION_FILE, "r", encoding="utf-8") as file:
        report = json.load(file)

    unanswerable = [
        record
        for record in report["query_scores"]
        if not record["answerable"]
    ]

    passed = 0
    rejected = 0

    for record in unanswerable:

        score = record["max_vector_score"]

        accepted = (
            score is not None
            and score >= THRESHOLD
        )

        if accepted:
            passed += 1
        else:
            rejected += 1

        print("\n" + "-" * 65)

        print(f"Case ID: {record['case_id']}")
        print(f"Category: {record['category']}")
        print(f"Query: {record['query']}")
        print(f"Maximum vector score: {score}")
        print(f"Retrieved incidents: {record['retrieved_incident_ids']}")

        print(
            "Threshold decision:",
            "ACCEPTED" if accepted else "REJECTED"
        )

    print("\n========== SUMMARY ==========")
    print(f"Total unanswerable queries: {len(unanswerable)}")
    print(f"Incorrectly accepted: {passed}")
    print(f"Correctly rejected: {rejected}")


if __name__ == "__main__":
    main()