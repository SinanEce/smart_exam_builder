from __future__ import annotations

import json
import re
from typing import Any


class JSONExtractionError(ValueError):
    """Raised when a model response cannot be converted into JSON."""


def extract_json_text(raw_text: str) -> str:
    """Extract the first plausible JSON object or array from a text response."""
    text = raw_text.strip()
    if not text:
        raise JSONExtractionError("Empty LLM response.")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    if text.startswith("{") or text.startswith("["):
        return text

    object_start = text.find("{")
    array_start = text.find("[")
    starts = [idx for idx in (object_start, array_start) if idx != -1]
    if not starts:
        raise JSONExtractionError("No JSON object or array found in LLM response.")

    start = min(starts)
    end_char = "}" if text[start] == "{" else "]"
    end = text.rfind(end_char)
    if end <= start:
        raise JSONExtractionError("Could not find the end of the JSON payload.")
    return text[start : end + 1]


def loads_json_from_text(raw_text: str) -> Any:
    return json.loads(extract_json_text(raw_text))


def compact_preview(text: str, max_chars: int = 260) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."

