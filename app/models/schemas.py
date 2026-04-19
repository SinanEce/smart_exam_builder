from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.utils.ids import short_uuid
from app.utils.validators import compact_preview


class CognitiveLevel(str, Enum):
    remember = "remember"
    understand = "understand"
    apply = "apply"
    analyze = "analyze"
    evaluate = "evaluate"


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class QuestionType(str, Enum):
    multiple_choice = "multiple_choice"
    open_ended = "open_ended"


class LearningOutcome(BaseModel):
    id: str = Field(..., examples=["LO1"])
    course: str = Field(..., examples=["Computer Networks"])
    text: str = Field(..., min_length=5)
    cognitive_level: CognitiveLevel = CognitiveLevel.apply
    topic: str = Field(..., min_length=2)


class TextChunk(BaseModel):
    chunk_id: str
    text: str = Field(..., min_length=1)
    source_file: str
    course: str
    topic: str | None = None
    page_number: int | None = None
    chunk_index: int = Field(..., ge=0)


class ChunkPreview(BaseModel):
    chunk_id: str
    source_file: str
    course: str
    topic: str | None = None
    page_number: int | None = None
    chunk_index: int
    preview_text: str

    @classmethod
    def from_chunk(cls, chunk: TextChunk) -> "ChunkPreview":
        return cls(
            chunk_id=chunk.chunk_id,
            source_file=chunk.source_file,
            course=chunk.course,
            topic=chunk.topic,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            preview_text=compact_preview(chunk.text),
        )


class RetrievedChunk(ChunkPreview):
    score: float = Field(..., description="Similarity score from the vector store.")

    @classmethod
    def from_chunk(cls, chunk: TextChunk, score: float) -> "RetrievedChunk":
        preview = ChunkPreview.from_chunk(chunk)
        return cls(**preview.model_dump(), score=score)


class MCQOption(BaseModel):
    label: Literal["A", "B", "C", "D"]
    text: str = Field(..., min_length=1)
    is_correct: bool = False


class GeneratedQuestion(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    question_id: str = Field(default_factory=lambda: short_uuid("q"))
    learning_outcome_id: str
    question_type: QuestionType
    difficulty: Difficulty
    question_text: str = Field(..., min_length=10)
    answer_key: str = Field(..., min_length=1)
    explanation: str = Field(..., min_length=5)
    source_chunks: list[str] = Field(default_factory=list)
    options: list[MCQOption] | None = None

    @model_validator(mode="after")
    def validate_question_shape(self) -> Self:
        if self.question_type in (QuestionType.multiple_choice, QuestionType.multiple_choice.value):
            if self.options is None or len(self.options) != 4:
                raise ValueError("Multiple-choice questions must include exactly 4 options.")
            labels = [option.label for option in self.options]
            if labels != ["A", "B", "C", "D"]:
                raise ValueError("Multiple-choice options must be labeled A, B, C, D.")
            correct_count = sum(1 for option in self.options if option.is_correct)
            if correct_count != 1:
                raise ValueError("Multiple-choice questions must have exactly one correct option.")
            correct_label = next(option.label for option in self.options if option.is_correct)
            if correct_label not in self.answer_key:
                raise ValueError("Answer key must mention the correct option label.")
        else:
            if self.options:
                raise ValueError("Open-ended questions must not include MCQ options.")
        return self


class QuestionSet(BaseModel):
    questions: list[GeneratedQuestion] = Field(..., min_length=1)


class CritiqueResponse(BaseModel):
    accepted: bool
    score: int = Field(..., ge=0, le=10)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    revised_focus: str | None = None


class AgentTraceEvent(BaseModel):
    agent: str
    action: str
    round: int = Field(..., ge=0)
    summary: str
    accepted: bool | None = None
    issues: list[str] = Field(default_factory=list)


class RubricCriterion(BaseModel):
    criterion_id: str
    description: str = Field(..., min_length=5)
    points: float = Field(..., gt=0)


class Rubric(BaseModel):
    rubric_id: str = Field(default_factory=lambda: short_uuid("rubric"))
    question_id: str
    criteria: list[RubricCriterion] = Field(..., min_length=1)
    total_points: float = Field(..., gt=0)

    @model_validator(mode="after")
    def validate_points(self) -> Self:
        total = round(sum(item.points for item in self.criteria), 2)
        if abs(total - self.total_points) > 0.01:
            raise ValueError("Rubric criterion points must sum to total_points.")
        return self


class GradingCriterionScore(BaseModel):
    criterion_id: str
    awarded_points: float = Field(..., ge=0)
    max_points: float = Field(..., gt=0)
    justification: str = Field(..., min_length=3)

    @model_validator(mode="after")
    def validate_award(self) -> Self:
        if self.awarded_points > self.max_points:
            raise ValueError("Awarded points cannot exceed max_points.")
        return self


class GradingResult(BaseModel):
    question_id: str
    rubric_id: str
    total_score: float = Field(..., ge=0)
    total_points: float = Field(..., gt=0)
    criterion_scores: list[GradingCriterionScore] = Field(..., min_length=1)
    feedback: str = Field(..., min_length=5)

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        total = round(sum(item.awarded_points for item in self.criterion_scores), 2)
        if abs(total - self.total_score) > 0.05:
            raise ValueError("Criterion scores must sum to total_score.")
        if self.total_score > self.total_points:
            raise ValueError("total_score cannot exceed total_points.")
        return self


class IngestionResponse(BaseModel):
    course: str
    indexed_chunks: int
    total_chunks_in_store: int
    source_files: list[str]
    chunks: list[ChunkPreview]


class GenerateQuestionsRequest(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    learning_outcome: LearningOutcome | None = None
    learning_outcome_id: str | None = Field(default="LO1")
    course: str = "Computer Networks"
    difficulty: Difficulty = Difficulty.medium
    question_type: QuestionType = QuestionType.multiple_choice
    question_count: int = Field(default=3, ge=1, le=10)
    top_k: int = Field(default=5, ge=1, le=12)
    max_revision_rounds: int = Field(default=2, ge=0, le=3)

    @model_validator(mode="after")
    def validate_learning_outcome_reference(self) -> Self:
        if self.learning_outcome is None and not self.learning_outcome_id:
            raise ValueError("Provide learning_outcome or learning_outcome_id.")
        return self


class GenerateQuestionsResponse(BaseModel):
    learning_outcome: LearningOutcome
    retrieved_chunks: list[RetrievedChunk]
    questions: list[GeneratedQuestion]
    writer_critic_trace: list[AgentTraceEvent]


class GenerateRubricRequest(BaseModel):
    question: GeneratedQuestion
    total_points: float = Field(default=10, gt=0, le=100)


class GenerateRubricResponse(BaseModel):
    question: GeneratedQuestion
    rubric: Rubric


class GradeAnswerRequest(BaseModel):
    question: GeneratedQuestion
    rubric: Rubric
    student_answer: str = Field(..., min_length=1)


class GradeAnswerResponse(BaseModel):
    result: GradingResult


class HealthResponse(BaseModel):
    status: str
    app_name: str
    vector_store_chunks: int
    mock_llm_enabled: bool


class ErrorResponse(BaseModel):
    detail: str


def model_schemas() -> dict[str, dict[str, Any]]:
    exposed = [
        LearningOutcome,
        TextChunk,
        RetrievedChunk,
        GeneratedQuestion,
        Rubric,
        GradingResult,
        GenerateQuestionsRequest,
        GenerateQuestionsResponse,
    ]
    return {model.__name__: model.model_json_schema() for model in exposed}
