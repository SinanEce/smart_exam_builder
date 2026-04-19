from __future__ import annotations

from app.agents.grader import GraderAgent
from app.agents.rubric_agent import RubricAgent
from app.core.llm_client import MockLLMClient
from app.models.schemas import GenerateRubricRequest, GeneratedQuestion, GradeAnswerRequest
from app.services.grading_service import GradingService
from app.services.rubric_service import RubricService


def test_rubric_and_grading_structures() -> None:
    question = GeneratedQuestion(
        learning_outcome_id="LO3",
        question_type="open_ended",
        difficulty="medium",
        question_text="DHCP DORA sürecini açıklayınız.",
        answer_key="Discover, Offer, Request ve Acknowledge adımları açıklanmalıdır.",
        explanation="Soru DHCP sürecinin adımlarını ölçer.",
        source_chunks=["chunk_dhcp"],
    )
    llm = MockLLMClient()
    rubric_response = RubricService(RubricAgent(llm)).generate(GenerateRubricRequest(question=question, total_points=10))

    assert rubric_response.rubric.total_points == 10
    assert len(rubric_response.rubric.criteria) == 3

    grade_response = GradingService(GraderAgent(llm)).grade(
        GradeAnswerRequest(
            question=question,
            rubric=rubric_response.rubric,
            student_answer=(
                "DHCP istemciye otomatik IP verir. Discover ile arama yapılır, Offer ile teklif gelir, "
                "Request ile istemci kabul eder ve Acknowledge ile sunucu onaylar."
            ),
        )
    )

    assert grade_response.result.total_score <= grade_response.result.total_points
    assert len(grade_response.result.criterion_scores) == 3
    assert grade_response.result.feedback

