from src.pipeline.root_cause_pipeline import RootCausePipeline


def main():

    print("=" * 60)
    print("             ROOTCAUSE AI")
    print("       SOFTWARE INCIDENT INVESTIGATION")
    print("=" * 60)

    pipeline = RootCausePipeline(
        data_directory="data/raw",
        persist_directory="vectorstore/first_investigation",
        model_name="gpt-4o-mini"
    )

    query = (
        "Why is the Payment Service returning HTTP 500 "
        "errors during payment processing?"
    )

    result = pipeline.run(
        query=query,
        top_k=3,
        candidate_k=5
    )

    print("\n" + "=" * 60)
    print("RETRIEVED HISTORICAL EVIDENCE")
    print("=" * 60)

    for index, evidence in enumerate(
        result["retrieved_evidence"],
        start=1
    ):

        print(f"\nEvidence {index}")
        print("Incident ID:", evidence["incident_id"])
        print("RRF Score:", evidence["rrf_score"])
        print("Content:", evidence["content"])

    print("\n" + "=" * 60)
    print("ROOT CAUSE INVESTIGATION REPORT")
    print("=" * 60)

    print(result["analysis"])

    print("\n" + "=" * 60)
    print("INVESTIGATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
