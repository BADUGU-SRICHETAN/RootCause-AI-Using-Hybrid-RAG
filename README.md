# RootCause AI
### Hybrid RAG-Based Software Incident Investigation System

RootCause AI is a Hybrid Retrieval-Augmented Generation (RAG) system designed to investigate software incidents using historical incident records.

It combines keyword-based retrieval, semantic search, and Large Language Models (LLMs) to retrieve relevant historical incidents and generate evidence-grounded root-cause hypotheses.

The system incorporates query-scope validation and evidence-based abstention to reduce unsupported responses.

---

## 🚀 Key Features

- **Hybrid Retrieval:** Combines BM25 keyword retrieval and vector similarity search.
- **Reciprocal Rank Fusion (RRF):** Combines rankings from multiple retrieval methods.
- **Evidence-Grounded Analysis:** Generates root-cause hypotheses based on retrieved incident records.
- **Evidence-Based Abstention:** Avoids generating conclusions when supporting evidence is insufficient.
- **Query Scope Validation:** Identifies questions outside the system's intended investigation scope.
- **Incident References:** Provides supporting incident records and retrieval information.
- **Retrieval Evaluation:** Compares BM25, vector, and hybrid retrieval performance.
- **REST API:** FastAPI backend for incident investigation.
- **Interactive Dashboard:** Streamlit frontend for submitting and reviewing investigations.

---

## 🏗️ System Architecture

```text
                  User Query
                      |
                      v
             Query Scope Validation
                      |
                      v
                Hybrid Retrieval
                      |
              +-------+-------+
              |               |
              v               v
         BM25 Search      Vector Search
              |               |
              +-------+-------+
                      |
                      v
           Reciprocal Rank Fusion
                      |
                      v
              Evidence Evaluation
                      |
               +------+------+
               |             |
               v             v
          Sufficient    Insufficient
           Evidence      Evidence
               |             |
               v             v
         Root-Cause      Abstention
          Analysis        Response
               |
               v
       Evidence-Grounded Result
               |
               v
          FastAPI Backend
               |
               v
        Streamlit Dashboard