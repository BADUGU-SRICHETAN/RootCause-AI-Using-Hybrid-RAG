from fastapi.testclient import TestClient

import src.api.main as api_module


client = TestClient(api_module.app)


class FakePipeline:

    def run(self, query, top_k, candidate_k):

        return {
            "query": query,
            "retrieved_count": 1,
            "retrieved_evidence": [
                {
                    "incident_id": "INC001",
                    "rrf_score": 0.0327
                }
            ],
            "analysis": "Database connection pool exhaustion."
        }


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_investigation_endpoint(monkeypatch):

    monkeypatch.setattr(
        api_module,
        "get_pipeline",
        lambda: FakePipeline()
    )

    response = client.post(
        "/investigate",
        json={
            "query": "Why is Payment Service returning HTTP 500 errors?",
            "top_k": 3,
            "candidate_k": 5
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"

    assert (
        data["investigation"]["retrieved_evidence"][0]["incident_id"]
        == "INC001"
    )


def test_invalid_candidate_k():

    response = client.post(
        "/investigate",
        json={
            "query": "Why is Payment Service returning HTTP 500 errors?",
            "top_k": 5,
            "candidate_k": 2
        }
    )

    assert response.status_code == 422


def test_empty_query():

    response = client.post(
        "/investigate",
        json={
            "query": "",
            "top_k": 3,
            "candidate_k": 5
        }
    )

    assert response.status_code == 422