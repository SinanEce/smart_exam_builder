from __future__ import annotations

import json
from pathlib import Path

from app.core.llm_client import LLMClient
from app.models.schemas import CritiqueResponse, LearningOutcome, QuestionSet, RetrievedChunk


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "critic_prompt.txt"


class CriticAgent:
    """Reviews question drafts and returns accept/revise guidance."""

    def __init__(self, llm_client: LLMClient, prompt_path: Path = PROMPT_PATH) -> None:
        self.llm_client = llm_client
        self.prompt_template = prompt_path.read_text(encoding="utf-8")

    def review(
        self,
        learning_outcome: LearningOutcome,
        retrieved_chunks: list[RetrievedChunk],
        questions: QuestionSet,
        round_index: int,
    ) -> CritiqueResponse:
        payload = {
            "learning_outcome": learning_outcome.model_dump(mode="json"),
            "retrieved_chunks": [chunk.model_dump(mode="json") for chunk in retrieved_chunks],
            "questions": [question.model_dump(mode="json") for question in questions.questions],
            "round": round_index,
            "output_language": "Turkish",
        }
        prompt = self.prompt_template.replace("{{request_json}}", json.dumps(payload, ensure_ascii=False, indent=2))
        return self.llm_client.generate_structured(prompt, CritiqueResponse)

