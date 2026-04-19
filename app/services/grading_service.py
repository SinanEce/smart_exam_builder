from __future__ import annotations

from app.agents.grader import GraderAgent
from app.models.schemas import GradeAnswerRequest, GradeAnswerResponse


class GradingService:
    def __init__(self, grader_agent: GraderAgent) -> None:
        self.grader_agent = grader_agent

    def grade(self, request: GradeAnswerRequest) -> GradeAnswerResponse:
        result = self.grader_agent.grade(
            question=request.question,
            rubric=request.rubric,
            student_answer=request.student_answer,
        )
        return GradeAnswerResponse(result=result)

