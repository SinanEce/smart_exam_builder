from __future__ import annotations

import json
from pathlib import Path

from app.core.llm_client import LLMClient
from app.models.schemas import GeneratedQuestion, GradingResult, Rubric


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "grader_prompt.txt"


class GraderAgent:
    """Grades a student answer with a generated rubric."""

    def __init__(self, llm_client: LLMClient, prompt_path: Path = PROMPT_PATH) -> None:
        self.llm_client = llm_client
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    def grade(self, question: GeneratedQuestion, rubric: Rubric, student_answer: str) -> GradingResult:
        payload = {
            "question": question.model_dump(mode="json"),
            "rubric": rubric.model_dump(mode="json"),
            "student_answer": student_answer,
            "output_language": "Turkish",
        }
        prompt = self.prompt_template.replace("{{request_json}}", json.dumps(payload, ensure_ascii=False, indent=2))
        return self.llm_client.generate_structured(prompt, GradingResult)

