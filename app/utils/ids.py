from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime


def short_uuid(prefix: str) -> str:
    """Return a compact, URL-safe id with a stable prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def stable_id(prefix: str, *parts: object, length: int = 14) -> str:
    """Return a deterministic id from source metadata."""
    joined = "::".join(str(part) for part in parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()

