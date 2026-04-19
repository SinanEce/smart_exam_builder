from __future__ import annotations

from functools import lru_cache

from app.agents.critic import CriticAgent
from app.agents.grader import GraderAgent
from app.agents.retriever import RetrieverAgent
from app.agents.rubric_agent import RubricAgent
from app.agents.writer import WriterAgent
from app.core.config import Settings, get_settings
from app.core.llm_client import LLMClient, create_llm_client
from app.rag.embeddings import HashingEmbeddingModel
from app.rag.text_chunker import TextChunker
from app.rag.vector_store import VectorStore
from app.services.grading_service import GradingService
from app.services.ingestion_service import IngestionService
from app.services.question_service import QuestionService
from app.services.rag_service import RagService
from app.services.rubric_service import RubricService


@lru_cache(maxsize=1)
def get_embedding_model() -> HashingEmbeddingModel:
    settings = get_settings()
    return HashingEmbeddingModel(dimension=settings.embedding_dim)


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    settings = get_settings()
    assert settings.processed_data_dir is not None
    return VectorStore(storage_dir=settings.processed_data_dir, dimension=settings.embedding_dim)


@lru_cache(maxsize=1)
def get_llm_client() -> LLMClient:
    return create_llm_client(get_settings())


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        chunker=TextChunker(max_chars=settings.max_chunk_chars, overlap_chars=settings.chunk_overlap_chars),
        embedding_model=get_embedding_model(),
        vector_store=get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_rag_service() -> RagService:
    return RagService(embedding_model=get_embedding_model(), vector_store=get_vector_store())


@lru_cache(maxsize=1)
def get_question_service() -> QuestionService:
    settings = get_settings()
    assert settings.raw_data_dir is not None
    assert settings.samples_dir is not None
    llm_client = get_llm_client()
    rag_service = get_rag_service()
    return QuestionService(
        retriever_agent=RetrieverAgent(rag_service),
        writer_agent=WriterAgent(llm_client),
        critic_agent=CriticAgent(llm_client),
        ingestion_service=get_ingestion_service(),
        raw_data_dir=settings.raw_data_dir,
        samples_dir=settings.samples_dir,
    )


@lru_cache(maxsize=1)
def get_rubric_service() -> RubricService:
    return RubricService(rubric_agent=RubricAgent(get_llm_client()))


@lru_cache(maxsize=1)
def get_grading_service() -> GradingService:
    return GradingService(grader_agent=GraderAgent(get_llm_client()))


def settings_dependency() -> Settings:
    return get_settings()

