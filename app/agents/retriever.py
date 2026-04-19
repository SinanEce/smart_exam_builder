from __future__ import annotations

from app.models.schemas import Difficulty, LearningOutcome, QuestionType, RetrievedChunk
from app.rag.retrieval import build_retrieval_query
from app.services.rag_service import RagService


class RetrieverAgent:
    """Retrieves course chunks relevant to a learning outcome."""

    def __init__(self, rag_service: RagService) -> None:
        self.rag_service = rag_service

    def retrieve(
        self,
        learning_outcome: LearningOutcome,
        question_type: QuestionType | str,
        difficulty: Difficulty | str,
        top_k: int,
    ) -> list[RetrievedChunk]:
        query = build_retrieval_query(learning_outcome, question_type, difficulty)
        return self.rag_service.search(query=query, course=learning_outcome.course, top_k=top_k)

