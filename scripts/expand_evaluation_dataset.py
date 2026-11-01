import json
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

EVALUATION_FILE = (
    PROJECT_ROOT / "data" / "evaluation" / "retrieval_test_cases.json"
)

INCIDENT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "incidents.json"
)


# --------------------------------------------------
# 1. PARAPHRASED QUERIES (20)
# --------------------------------------------------

PARAPHRASED = [
    ("How can insufficient database connections cause payment failures?",
     ["INC001"]),

    ("What makes order history database searches exceed their time limit?",
     ["INC002"]),

    ("Why does Inventory Service fail to log in to its database after a password change?",
     ["INC003"]),

    ("What could cause reporting jobs to consume all available database sessions?",
     ["INC004"]),

    ("Why do simultaneous checkout operations block one another?",
     ["INC005"]),

    ("What can make a database read replica fall behind the primary database?",
     ["INC006"]),

    ("Why does User Service fail to reconnect when the database comes back online?",
     ["INC007"]),

    ("What happens when temporary database files consume all available disk space?",
     ["INC008"]),

    ("Why might billing transactions take unusually long to commit?",
     ["INC009"]),

    ("What causes background notification workers to hold database connections indefinitely?",
     ["INC010"]),

    ("Why does Customer API crash when an address field is null?",
     ["INC011"]),

    ("What makes Search API requests wait too long for the recommendation service?",
     ["INC012"]),

    ("Why does the API return an internal error for malformed JSON input?",
     ["INC013"]),

    ("Why are API clients being throttled even when their request rates are low?",
     ["INC014"]),

    ("What configuration prevents browser OPTIONS requests from passing through the gateway?",
     ["INC015"]),

    ("Why does Product API fail when a requested page contains fewer items than expected?",
     ["INC016"]),

    ("What causes JSON serialization to fail when an event contains a Java timestamp?",
     ["INC017"]),

    ("Why does Shipping API expose connection failures from an unavailable dependency?",
     ["INC018"]),

    ("Why are document uploads rejected when their size is within the documented limit?",
     ["INC019"]),

    ("What prevents Admin API clients from updating resources using PATCH?",
     ["INC020"]),
]


# --------------------------------------------------
# 2. AMBIGUOUS QUERIES (15)
# --------------------------------------------------

AMBIGUOUS = [
    ("Why is the application unable to obtain a connection?",
     ["INC001", "INC004", "INC010"]),

    ("Why are database operations becoming slow?",
     ["INC002", "INC006", "INC009"]),

    ("Why does the application fail to connect after a system change?",
     ["INC003", "INC007"]),

    ("Why are requests failing during checkout?",
     ["INC001", "INC005", "INC032", "INC040"]),

    ("Why is the service running out of resources?",
     ["INC008", "INC021", "INC027", "INC037"]),

    ("Why are API requests taking too long?",
     ["INC012", "INC022", "INC026", "INC029"]),

    ("Why is the application returning unexpected HTTP errors?",
     ["INC013", "INC016", "INC018", "INC020"]),

    ("Why are background jobs not completing successfully?",
     ["INC010", "INC023", "INC027"]),

    ("Why does the application fail during startup or deployment?",
     ["INC025", "INC032", "INC036", "INC038"]),

    ("Why are authentication requests being rejected?",
     ["INC003", "INC041", "INC043", "INC046"]),

    ("Why are users experiencing session or token problems?",
     ["INC042", "INC045", "INC048", "INC050"]),

    ("Why is the application not enforcing access restrictions correctly?",
     ["INC044", "INC049"]),

    ("Why are external service connections failing?",
     ["INC012", "INC018", "INC043"]),

    ("Why is the production environment behaving differently from expected?",
     ["INC031", "INC035", "INC040", "INC048"]),

    ("Why is the system experiencing storage or disk-related problems?",
     ["INC008", "INC030"]),
]


# --------------------------------------------------
# 3. MULTI-RELEVANT QUERIES (15)
# --------------------------------------------------

MULTI_RELEVANT = [
    ("Compare historical incidents caused by database connection leaks or insufficient connection capacity.",
     ["INC001", "INC004", "INC010"]),

    ("Which incidents involved database connectivity failures caused by credentials, recovery behavior, or connection configuration?",
     ["INC003", "INC007"]),

    ("Which historical incidents involved database performance problems caused by inefficient queries, replication lag, or locking?",
     ["INC002", "INC006", "INC009"]),

    ("Find incidents where exhausted memory or unbounded data structures affected application stability.",
     ["INC021", "INC027", "INC037"]),

    ("Which incidents involved CPU, garbage collection, or thread contention affecting application performance?",
     ["INC022", "INC023", "INC024"]),

    ("Find historical API incidents involving slow dependencies, request timeouts, or large response processing costs.",
     ["INC012", "INC026"]),

    ("Which incidents involved HTTP errors caused by application request handling or incorrect API behavior?",
     ["INC013", "INC016", "INC020"]),

    ("Find deployment incidents involving missing configuration or incorrect runtime settings.",
     ["INC031", "INC034", "INC036"]),

    ("Which incidents were caused by incorrect deployment artifacts or deployment sequencing?",
     ["INC032", "INC033"]),

    ("Compare authentication incidents involving invalid signing keys, expired tokens, and incorrect token validation.",
     ["INC041", "INC042", "INC050"]),

    ("Which incidents involved incorrect authentication or authorization configuration?",
     ["INC041", "INC046", "INC048"]),

    ("Find incidents where security controls failed to protect access or prevent abusive login attempts.",
     ["INC044", "INC049"]),

    ("Which incidents involved disk capacity, disk I/O, or storage-related resource pressure?",
     ["INC008", "INC030"]),

    ("Compare incidents where connection pools or worker threads became bottlenecks under concurrent workloads.",
     ["INC001", "INC023", "INC028"]),

    ("Which incidents involved failures in distributed service communication or external dependency connectivity?",
     ["INC012", "INC018", "INC043"]),
]


# --------------------------------------------------
# 4. UNANSWERABLE QUERIES (10)
# --------------------------------------------------

UNANSWERABLE = [
    "What is the current monthly cloud hosting cost of the Payment Service?",

    "Who is the engineer responsible for fixing incident INC001?",

    "Which production server currently hosts the Order Service?",

    "What will be the exact number of API requests tomorrow?",

    "Which monitoring vendor does the organization currently purchase?",

    "What is the annual salary of the database administrator?",

    "What is the current physical location of the production data center?",

    "Which employee approved the January 2026 deployment?",

    "What is the exact hardware purchase price for the database servers?",

    "Predict the precise date of the next production outage.",
]


# --------------------------------------------------
# 5. LOAD EXISTING DATASETS
# --------------------------------------------------

with open(EVALUATION_FILE, "r", encoding="utf-8") as file:
    original_cases = json.load(file)

with open(INCIDENT_FILE, "r", encoding="utf-8") as file:
    incidents = json.load(file)


incident_ids = {
    incident["incident_id"]
    for incident in incidents
}


# --------------------------------------------------
# 6. NORMALIZE ORIGINAL CASES
# --------------------------------------------------

expanded_cases = []

for index, case in enumerate(original_cases, start=1):

    expanded_cases.append({
        "case_id": f"ORIG{index:03d}",
        "category": "original",
        "query": case["query"],
        "answerable": True,
        "relevant_incident_ids": case["relevant_incident_ids"]
    })


# --------------------------------------------------
# 7. ADD PARAPHRASED CASES
# --------------------------------------------------

for index, (query, relevant_ids) in enumerate(
    PARAPHRASED,
    start=1
):

    expanded_cases.append({
        "case_id": f"PARA{index:03d}",
        "category": "paraphrased",
        "query": query,
        "answerable": True,
        "relevant_incident_ids": relevant_ids
    })


# --------------------------------------------------
# 8. ADD AMBIGUOUS CASES
# --------------------------------------------------

for index, (query, relevant_ids) in enumerate(
    AMBIGUOUS,
    start=1
):

    expanded_cases.append({
        "case_id": f"AMB{index:03d}",
        "category": "ambiguous",
        "query": query,
        "answerable": True,
        "relevant_incident_ids": relevant_ids
    })


# --------------------------------------------------
# 9. ADD MULTI-RELEVANT CASES
# --------------------------------------------------

for index, (query, relevant_ids) in enumerate(
    MULTI_RELEVANT,
    start=1
):

    expanded_cases.append({
        "case_id": f"MULTI{index:03d}",
        "category": "multi_relevant",
        "query": query,
        "answerable": True,
        "relevant_incident_ids": relevant_ids
    })


# --------------------------------------------------
# 10. ADD UNANSWERABLE CASES
# --------------------------------------------------

for index, query in enumerate(
    UNANSWERABLE,
    start=1
):

    expanded_cases.append({
        "case_id": f"UNANS{index:03d}",
        "category": "unanswerable",
        "query": query,
        "answerable": False,
        "relevant_incident_ids": []
    })


# --------------------------------------------------
# 11. VALIDATE DATASET
# --------------------------------------------------

case_ids = [
    case["case_id"]
    for case in expanded_cases
]

assert len(case_ids) == len(set(case_ids)), (
    "Duplicate case IDs detected."
)

for case in expanded_cases:

    assert case["query"].strip(), (
        f"Empty query: {case['case_id']}"
    )

    if case["answerable"]:

        assert case["relevant_incident_ids"], (
            f"Missing relevant IDs: {case['case_id']}"
        )

        for incident_id in case["relevant_incident_ids"]:

            assert incident_id in incident_ids, (
                f"Unknown incident ID {incident_id} "
                f"in {case['case_id']}"
            )

    else:

        assert not case["relevant_incident_ids"], (
            f"Unanswerable case has relevant IDs: "
            f"{case['case_id']}"
        )


# --------------------------------------------------
# 12. SAVE EXPANDED DATASET
# --------------------------------------------------

with open(EVALUATION_FILE, "w", encoding="utf-8") as file:

    json.dump(
        expanded_cases,
        file,
        indent=4,
        ensure_ascii=False
    )


# --------------------------------------------------
# 13. DISPLAY SUMMARY
# --------------------------------------------------

category_counts = Counter(
    case["category"]
    for case in expanded_cases
)

print("\n======================================")
print(" ROOTCAUSE AI EVALUATION DATASET")
print("======================================")

print("Total test cases:", len(expanded_cases))

print("\nCategory distribution:")

for category, count in category_counts.items():
    print(f"{category:20s}: {count}")

print("\nAnswerable:", sum(
    case["answerable"]
    for case in expanded_cases
))

print("Unanswerable:", sum(
    not case["answerable"]
    for case in expanded_cases
))

print("\nAll validation checks passed.")
print("Evaluation dataset updated successfully.")
print("File:", EVALUATION_FILE)