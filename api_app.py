import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent.fitness_workflow import build_fitness_workflow
from domain.training import TrainingRecordCreate
from infrastructure.training_repository import SQLiteTrainingRepository
from rag.evaluable_retriever import EvaluableHybridRetriever


class AgentRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    user_id: str = Field(default="anonymous", min_length=1, max_length=64)


def create_app(repository=None, retriever=None) -> FastAPI:
    root = Path(__file__).resolve().parent
    repository = repository or SQLiteTrainingRepository(root / "data" / "training.db")
    retriever = retriever or EvaluableHybridRetriever.from_directory(root / "data")
    workflow = build_fitness_workflow(retriever)
    app = FastAPI(title="智练通健身 Agent", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "zhiliantong-agent", "version": "1.0.0"}

    @app.post("/v1/training-records", status_code=201)
    def create_record(command: TrainingRecordCreate, idempotency_key: str | None = Header(None)):
        if idempotency_key and idempotency_key != command.request_id:
            raise HTTPException(400, "Idempotency-Key 必须与 request_id 一致")
        record, created = repository.save(command)
        return {"created": created, "record": record.model_dump(mode="json")}

    @app.get("/v1/users/{user_id}/summaries/{month}")
    def summary(user_id: str, month: str):
        return repository.monthly_summary(user_id, month)

    @app.post("/v1/agent/run")
    def run_agent(request: AgentRequest):
        return workflow.invoke(
            {"query": request.query},
            config={"configurable": {"thread_id": f"{request.user_id}-{uuid.uuid4()}"}},
        )

    @app.post("/v1/agent/stream")
    def stream_agent(request: AgentRequest):
        def events():
            result = run_agent(request)
            for step in result["steps"]:
                yield f"event: progress\ndata: {json.dumps({'step': step}, ensure_ascii=False)}\n\n"
            yield f"event: result\ndata: {json.dumps(result, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


app = create_app()
