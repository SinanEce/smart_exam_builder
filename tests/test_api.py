from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_generate_questions_smoke() -> None:
    client = TestClient(app)
    ingest_response = client.post("/ingest-materials", data={"course": "Computer Networks"})
    assert ingest_response.status_code == 200

    response = client.post(
        "/generate-questions",
        json={
            "learning_outcome_id": "LO1",
            "course": "Computer Networks",
            "difficulty": "medium",
            "question_type": "multiple_choice",
            "question_count": 1,
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieved_chunks"]
    assert body["questions"][0]["source_chunks"]
    assert body["writer_critic_trace"]

