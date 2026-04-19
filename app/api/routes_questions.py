from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.container import get_question_service
from app.models.schemas import GenerateQuestionsRequest, GenerateQuestionsResponse
from app.services.question_service import QuestionService

router = APIRouter(tags=["Questions"])


@router.post("/generate-questions", response_model=GenerateQuestionsResponse)
def generate_questions(
    request: GenerateQuestionsRequest,
    service: QuestionService = Depends(get_question_service),
) -> GenerateQuestionsResponse:
    try:
        return service.generate(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

