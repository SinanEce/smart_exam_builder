from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.container import get_grading_service
from app.models.schemas import GradeAnswerRequest, GradeAnswerResponse
from app.services.grading_service import GradingService

router = APIRouter(tags=["Grading"])


@router.post("/grade-answer", response_model=GradeAnswerResponse)
def grade_answer(
    request: GradeAnswerRequest,
    service: GradingService = Depends(get_grading_service),
) -> GradeAnswerResponse:
    return service.grade(request)

