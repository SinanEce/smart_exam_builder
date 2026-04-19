from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.schemas import GeneratedQuestion, MCQOption, QuestionType


def test_multiple_choice_requires_four_options_and_one_correct() -> None:
    question = GeneratedQuestion(
        learning_outcome_id="LO1",
        question_type=QuestionType.multiple_choice,
        difficulty="medium",
        question_text="IPv4 alt ağlama için doğru ifade hangisidir?",
        answer_key="A seçeneği doğrudur.",
        explanation="A seçeneği ağ ve host bitlerini doğru ilişkilendirir.",
        source_chunks=["chunk_1"],
        options=[
            MCQOption(label="A", text="Doğru seçenek", is_correct=True),
            MCQOption(label="B", text="Yanlış seçenek"),
            MCQOption(label="C", text="Yanlış seçenek"),
            MCQOption(label="D", text="Yanlış seçenek"),
        ],
    )

    assert question.options is not None
    assert sum(option.is_correct for option in question.options) == 1


def test_multiple_choice_rejects_two_correct_options() -> None:
    with pytest.raises(ValidationError):
        GeneratedQuestion(
            learning_outcome_id="LO1",
            question_type=QuestionType.multiple_choice,
            difficulty="medium",
            question_text="IPv4 alt ağlama için doğru ifade hangisidir?",
            answer_key="A seçeneği doğrudur.",
            explanation="A seçeneği ağ ve host bitlerini doğru ilişkilendirir.",
            source_chunks=["chunk_1"],
            options=[
                MCQOption(label="A", text="Doğru seçenek", is_correct=True),
                MCQOption(label="B", text="İkinci doğru", is_correct=True),
                MCQOption(label="C", text="Yanlış seçenek"),
                MCQOption(label="D", text="Yanlış seçenek"),
            ],
        )

