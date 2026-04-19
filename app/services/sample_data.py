from __future__ import annotations

import json
from pathlib import Path

from app.models.schemas import LearningOutcome


def load_learning_outcomes(samples_dir: Path) -> dict[str, LearningOutcome]:
    path = samples_dir / "learning_outcomes.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    outcomes = [LearningOutcome.model_validate(item) for item in raw]
    return {outcome.id: outcome for outcome in outcomes}

