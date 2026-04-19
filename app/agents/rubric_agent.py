from __future__ import annotations

import json
from pathlib import Path

from app.core.llm_client import LLMClient
from app.models.schemas import GeneratedQuestion, Rubric


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "rubric_prompt.txt"


class RubricAgent:
    """Generates analytic rubrics for open-ended questions."""

    def __init__(self, llm_client: LLMClient, prompt_path: Path = PROMPT_PATH) -> None:
        self.llm_client = llm_client
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    def generate(self, question: GeneratedQuestion, total_points: float) -> Rubric:
        payload = {
            "question": question.model_dump(mode="json"),
            "total_points": total_points,
            "output_language": "Turkish",
        }
        prompt = self.prompt_template.replace("{{request_json}}", json.dumps(payload, ensure_ascii=False, indent=2))
        return self.llm_client.generate_structured(prompt, Rubric)

