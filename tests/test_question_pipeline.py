from __future__ import annotations

from pathlib import Path

from app.agents.critic import CriticAgent
from app.agents.retriever import RetrieverAgent
from app.agents.writer import WriterAgent
from app.core.llm_client import MockLLMClient
from app.models.schemas import GenerateQuestionsRequest, LearningOutcome, TextChunk
from app.rag.embeddings import HashingEmbeddingModel
from app.rag.text_chunker import TextChunker
from app.rag.vector_store import VectorStore
from app.services.ingestion_service import IngestionService
from app.services.question_service import QuestionService
from app.services.rag_service import RagService


def test_question_generation_pipeline_returns_traceable_rag_output(workspace_tmp) -> None:
    embedding_model = HashingEmbeddingModel(dimension=128)
    vector_store = VectorStore(storage_dir=workspace_tmp / "processed", dimension=128)
    chunk = TextChunk(
        chunk_id="chunk_subnet",
        text="IPv4 alt ağlama CIDR maskesi ve host sayısı hesaplama için kullanılır.",
        source_file="subnet.md",
        course="Computer Networks",
        topic="IPv4 Alt Ağlama",
        chunk_index=0,
    )
    vector_store.upsert([chunk], embedding_model.embed_documents([chunk.text]))

    rag_service = RagService(embedding_model=embedding_model, vector_store=vector_store)
    llm = MockLLMClient()
    service = QuestionService(
        retriever_agent=RetrieverAgent(rag_service),
        writer_agent=WriterAgent(llm),
        critic_agent=CriticAgent(llm),
        ingestion_service=IngestionService(TextChunker(), embedding_model, vector_store),
        raw_data_dir=workspace_tmp / "raw",
        samples_dir=Path("unused"),
    )

    response = service.generate(
        GenerateQuestionsRequest(
            learning_outcome=LearningOutcome(
                id="LO1",
                course="Computer Networks",
                text="Öğrenci IPv4 alt ağlama işlemlerini yapabilir.",
                topic="IPv4 Alt Ağlama",
                cognitive_level="apply",
            ),
            learning_outcome_id=None,
            question_type="multiple_choice",
            question_count=2,
            top_k=1,
        )
    )

    assert len(response.questions) == 2
    assert response.retrieved_chunks[0].chunk_id == "chunk_subnet"
    assert response.questions[0].source_chunks == ["chunk_subnet"]
    assert [event.agent for event in response.writer_critic_trace] == ["Retriever", "Writer", "Critic"]
