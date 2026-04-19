from __future__ import annotations

from app.agents.rubric_agent import RubricAgent
from app.models.schemas import GenerateRubricRequest, GenerateRubricResponse, QuestionType


class RubricService:
    def __init__(self, rubric_agent: RubricAgent) -> None:
        self.rubric_agent = rubric_agent

    def generate(self, request: GenerateRubricRequest) -> GenerateRubricResponse:
        if request.question.question_type not in (QuestionType.open_ended, QuestionType.open_ended.value):
            raise ValueError("Rubrics are generated for open-ended questions.")
        rubric = self.rubric_agent.generate(question=request.question, total_points=request.total_points)
        return GenerateRubricResponse(question=request.question, rubric=rubric)

