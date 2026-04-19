from __future__ import annotations

from pathlib import Path

from app.agents.critic import CriticAgent
from app.agents.retriever import RetrieverAgent
from app.agents.writer import WriterAgent
from app.models.schemas import (
    AgentTraceEvent,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    LearningOutcome,
    QuestionSet,
)
from app.services.ingestion_service import IngestionService
from app.services.sample_data import load_learning_outcomes


class QuestionService:
    """Coordinates retriever, writer, and critic agents."""

    def __init__(
        self,
        retriever_agent: RetrieverAgent,
        writer_agent: WriterAgent,
        critic_agent: CriticAgent,
        ingestion_service: IngestionService,
        raw_data_dir: Path,
        samples_dir: Path,
    ) -> None:
        self.retriever_agent = retriever_agent
        self.writer_agent = writer_agent
        self.critic_agent = critic_agent
        self.ingestion_service = ingestion_service
        self.raw_data_dir = raw_data_dir
        self.samples_dir = samples_dir

    def generate(self, request: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
        learning_outcome = self._resolve_learning_outcome(request)
        retrieved = self.retriever_agent.retrieve(
            learning_outcome=learning_outcome,
            question_type=request.question_type,
            difficulty=request.difficulty,
            top_k=request.top_k,
        )

        if not retrieved:
            self.ingestion_service.ingest_local_folder(self.raw_data_dir, course=learning_outcome.course)
            retrieved = self.retriever_agent.retrieve(
                learning_outcome=learning_outcome,
                question_type=request.question_type,
                difficulty=request.difficulty,
                top_k=request.top_k,
            )

        if not retrieved:
            raise ValueError("No course material chunks are indexed. Ingest materials before generating questions.")

        trace: list[AgentTraceEvent] = [
            AgentTraceEvent(
                agent="Retriever",
                action="retrieve",
                round=0,
                summary=f"{len(retrieved)} kaynak parça getirildi.",
            )
        ]

        question_set = self.writer_agent.draft(
            learning_outcome=learning_outcome,
            retrieved_chunks=retrieved,
            difficulty=request.difficulty,
            question_type=request.question_type,
            question_count=request.question_count,
        )
        question_set = self._ensure_source_chunks(question_set, [chunk.chunk_id for chunk in retrieved])
        trace.append(
            AgentTraceEvent(
                agent="Writer",
                action="draft",
                round=0,
                summary=f"{len(question_set.questions)} soru taslağı oluşturuldu.",
            )
        )

        for round_index in range(request.max_revision_rounds + 1):
            critique = self.critic_agent.review(
                learning_outcome=learning_outcome,
                retrieved_chunks=retrieved,
                questions=question_set,
                round_index=round_index,
            )
            trace.append(
                AgentTraceEvent(
                    agent="Critic",
                    action="review",
                    round=round_index,
                    summary=f"Eleştiri puanı: {critique.score}/10.",
                    accepted=critique.accepted,
                    issues=critique.issues,
                )
            )
            if critique.accepted:
                break
            if round_index >= request.max_revision_rounds:
                break
            question_set = self.writer_agent.revise(
                learning_outcome=learning_outcome,
                retrieved_chunks=retrieved,
                difficulty=request.difficulty,
                question_type=request.question_type,
                question_count=request.question_count,
                previous_questions=question_set,
                critique=critique,
            )
            question_set = self._ensure_source_chunks(question_set, [chunk.chunk_id for chunk in retrieved])
            trace.append(
                AgentTraceEvent(
                    agent="Writer",
                    action="revise",
                    round=round_index + 1,
                    summary="Critic geri bildirimine göre sorular revize edildi.",
                )
            )

        return GenerateQuestionsResponse(
            learning_outcome=learning_outcome,
            retrieved_chunks=retrieved,
            questions=question_set.questions,
            writer_critic_trace=trace,
        )

    def _resolve_learning_outcome(self, request: GenerateQuestionsRequest) -> LearningOutcome:
        if request.learning_outcome is not None:
            return request.learning_outcome

        outcomes = load_learning_outcomes(self.samples_dir)
        if request.learning_outcome_id in outcomes:
            return outcomes[request.learning_outcome_id]
        raise ValueError(f"Learning outcome not found: {request.learning_outcome_id}")

    @staticmethod
    def _ensure_source_chunks(question_set: QuestionSet, fallback_chunk_ids: list[str]) -> QuestionSet:
        fallback = fallback_chunk_ids[:3]
        for question in question_set.questions:
            if not question.source_chunks:
                question.source_chunks = fallback
            else:
                question.source_chunks = [chunk_id for chunk_id in question.source_chunks if chunk_id in fallback_chunk_ids]
                if not question.source_chunks:
                    question.source_chunks = fallback
        return question_set

