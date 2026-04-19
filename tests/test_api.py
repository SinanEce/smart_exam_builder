from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_materials_openapi_shows_clean_file_upload_field() -> None:
    client = TestClient(app)
    response = client.get("/openapi.json")

    assert response.status_code == 200
    operation = response.json()["paths"]["/ingest-materials"]["post"]
    schema = operation["requestBody"]["content"]["multipart/form-data"]["schema"]
    properties = schema["properties"]

    assert properties["course"]["type"] == "string"
    assert properties["folder_path"]["type"] == "string"
    assert properties["files"]["type"] == "array"
    assert properties["files"]["items"] == {"type": "string", "format": "binary"}


def test_ingest_materials_folder_path_ignores_empty_files_form_value(workspace_tmp) -> None:
    materials_dir = workspace_tmp / "materials"
    materials_dir.mkdir(parents=True)
    (materials_dir / "dhcp_demo.txt").write_text(
        "DHCP DORA sureci Discover, Offer, Request ve Acknowledge adimlarindan olusur.",
        encoding="utf-8",
    )

    client = TestClient(app)
    response = client.post(
        "/ingest-materials",
        data={"course": "Computer Networks", "folder_path": str(materials_dir)},
        files={"files": (None, "")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["indexed_chunks"] == 1
    assert body["source_files"] == ["dhcp_demo.txt"]


def test_ingest_materials_accepts_real_uploaded_file() -> None:
    client = TestClient(app)
    response = client.post(
        "/ingest-materials",
        data={"course": "Computer Networks"},
        files={"files": ("uploaded_demo.txt", b"NAT port eslestirme ile genel IP paylasimi saglar.", "text/plain")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["indexed_chunks"] == 1
    assert body["source_files"] == ["uploaded_demo.txt"]


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
