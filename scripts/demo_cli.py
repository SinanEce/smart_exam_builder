from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.container import get_grading_service, get_ingestion_service, get_question_service, get_rubric_service  # noqa: E402
from app.models.schemas import GenerateQuestionsRequest, GenerateRubricRequest, GradeAnswerRequest  # noqa: E402


def main() -> None:
    settings = get_settings()
    assert settings.raw_data_dir is not None

    ingest_response = get_ingestion_service().ingest_local_folder(settings.raw_data_dir, course="Computer Networks")
    print("== Ingestion ==")
    print(ingest_response.model_dump_json(indent=2))

    question_response = get_question_service().generate(
        GenerateQuestionsRequest(
            learning_outcome_id="LO3",
            question_type="open_ended",
            difficulty="medium",
            question_count=1,
            top_k=4,
        )
    )
    print("\n== Question Generation ==")
    print(question_response.model_dump_json(indent=2))

    question = question_response.questions[0]
    rubric_response = get_rubric_service().generate(GenerateRubricRequest(question=question, total_points=10))
    print("\n== Rubric ==")
    print(rubric_response.model_dump_json(indent=2))

    grade_response = get_grading_service().grade(
        GradeAnswerRequest(
            question=question,
            rubric=rubric_response.rubric,
            student_answer=(
                "DHCP istemcinin otomatik IP almasını sağlar. İstemci Discover mesajı gönderir, sunucu Offer ile "
                "adres önerir, istemci Request ile ister ve sunucu Acknowledge ile kiralamayı onaylar. Ayrıca maske, "
                "varsayılan ağ geçidi ve DNS bilgileri de verilebilir."
            ),
        )
    )
    print("\n== Grading ==")
    print(json.dumps(grade_response.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

