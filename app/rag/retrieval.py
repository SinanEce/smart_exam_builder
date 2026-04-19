from __future__ import annotations

from app.models.schemas import Difficulty, LearningOutcome, QuestionType


def build_retrieval_query(
    learning_outcome: LearningOutcome,
    question_type: QuestionType | str,
    difficulty: Difficulty | str,
) -> str:
    return (
        f"{learning_outcome.course} {learning_outcome.topic}. "
        f"Öğrenme çıktısı: {learning_outcome.text}. "
        f"Soru tipi: {question_type}. Zorluk: {difficulty}."
    )

