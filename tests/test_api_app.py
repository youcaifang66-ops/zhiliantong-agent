from pathlib import Path

from fastapi.testclient import TestClient

from api_app import create_app
from infrastructure.training_repository import SQLiteTrainingRepository
from rag.evaluable_retriever import EvaluableHybridRetriever


ROOT = Path(__file__).resolve().parents[1]


def client():
    return TestClient(
        create_app(
            SQLiteTrainingRepository(), EvaluableHybridRetriever.from_directory(ROOT / "data")
        )
    )


def test_health_and_agent_contract():
    api = client()
    assert api.get("/health").json()["version"] == "1.0.0"
    response = api.post("/v1/agent/run", json={"user_id": "1001", "query": "新手每周练几次"})
    assert response.status_code == 200
    assert response.json()["intent"] == "plan"
    assert response.json()["citations"]


def test_training_record_idempotency_contract():
    api = client()
    payload = {
        "request_id": "request-api-001",
        "user_id": "1001",
        "training_date": "2026-09-21",
        "goal": "增肌",
        "sets": [{"exercise": "深蹲", "weight_kg": 60, "reps": 8, "rpe": 7}],
    }
    first = api.post("/v1/training-records", json=payload, headers={"Idempotency-Key": "request-api-001"})
    second = api.post("/v1/training-records", json=payload, headers={"Idempotency-Key": "request-api-001"})
    assert first.status_code == 201
    assert first.json()["created"] is True
    assert second.json()["created"] is False
