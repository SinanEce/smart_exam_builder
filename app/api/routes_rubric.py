from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.core.container import get_rubric_service
from app.models.schemas import GenerateRubricRequest, GenerateRubricResponse
from app.services.rubric_service import RubricService

router = APIRouter(tags=["Rubrics"])


@router.post("/generate-rubric", response_model=GenerateRubricResponse)
def generate_rubric(
    request: GenerateRubricRequest,
    service: RubricService = Depends(get_rubric_service),
) -> GenerateRubricResponse:
    try:
        return service.generate(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

