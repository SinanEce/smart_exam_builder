from __future__ import annotations

import json
from pathlib import Path

from app.core.llm_client import LLMClient
from app.models.schemas import (
    CritiqueResponse,
    Difficulty,
    LearningOutcome,
    QuestionSet,
    QuestionType,
    RetrievedChunk,
)


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "writer_prompt.txt"


class WriterAgent:
    """Drafts and revises questions from learning outcomes and retrieved chunks."""

    def __init__(self, llm_client: LLMClient, prompt_path: Path = PROMPT_PATH) -> None:
        self.llm_client = llm_client
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    def draft(
        self,
        learning_outcome: LearningOutcome,
        retrieved_chunks: list[RetrievedChunk],
        difficulty: Difficulty | str,
        question_type: QuestionType | str,
        question_count: int,
    ) -> QuestionSet:
        payload = self._payload(
            learning_outcome=learning_outcome,
            retrieved_chunks=retrieved_chunks,
            difficulty=difficulty,
            question_type=question_type,
            question_count=question_count,
            revision_notes=None,
            previous_questions=None,
        )
        return self.llm_client.generate_structured(self._prompt(payload), QuestionSet)

    def revise(
        self,
        learning_outcome: LearningOutcome,
        retrieved_chunks: list[RetrievedChunk],
        difficulty: Difficulty | str,
        question_type: QuestionType | str,
        question_count: int,
        previous_questions: QuestionSet,
        critique: CritiqueResponse,
    ) -> QuestionSet:
        payload = self._payload(
            learning_outcome=learning_outcome,
            retrieved_chunks=retrieved_chunks,
            difficulty=difficulty,
            question_type=question_type,
            question_count=question_count,
            revision_notes=critique.model_dump(mode="json"),
            previous_questions=previous_questions.model_dump(mode="json"),
        )
        return self.llm_client.generate_structured(self._prompt(payload), QuestionSet)

    def _prompt(self, payload: dict) -> str:
        request_json = json.dumps(payload, ensure_ascii=False, indent=2)
        return self.prompt_template.replace("{{request_json}}", request_json)

    @staticmethod
    def _payload(
        learning_outcome: LearningOutcome,
        retrieved_chunks: list[RetrievedChunk],
        difficulty: Difficulty | str,
        question_type: QuestionType | str,
        question_count: int,
        revision_notes: dict | None,
        previous_questions: dict | None,
    ) -> dict:
        return {
            "learning_outcome": learning_outcome.model_dump(mode="json"),
            "retrieved_chunks": [chunk.model_dump(mode="json") for chunk in retrieved_chunks],
            "difficulty": getattr(difficulty, "value", difficulty),
            "question_type": getattr(question_type, "value", question_type),
            "question_count": question_count,
            "revision_notes": revision_notes,
            "previous_questions": previous_questions,
            "output_language": "Turkish",
        }
