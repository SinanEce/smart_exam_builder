from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes_grading, routes_ingest, routes_questions, routes_rubric
from app.core.config import Settings
from app.core.container import get_vector_store, settings_dependency
from app.core.logging import setup_logging
from app.models.schemas import HealthResponse, model_schemas
from app.rag.vector_store import VectorStore

setup_logging()

app = FastAPI(
    title="SmartExam Builder",
    description="AI-powered Turkish exam generation, RAG traceability, rubrics, and grading.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_ingest.router)
app.include_router(routes_questions.router)
app.include_router(routes_rubric.router)
app.include_router(routes_grading.router)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health(
    settings: Settings = Depends(settings_dependency),
    vector_store: VectorStore = Depends(get_vector_store),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        vector_store_chunks=vector_store.count,
        mock_llm_enabled=settings.use_mock_llm or not bool(settings.openai_api_key),
    )


@app.get("/schemas", tags=["System"])
def schemas() -> dict:
    return model_schemas()

