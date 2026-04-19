from __future__ import annotations

import hashlib
import math
import re

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+")


class HashingEmbeddingModel:
    """A tiny deterministic embedding model for local demos and tests.

    It is intentionally dependency-light. The vector store can still use FAISS
    over these embeddings, and the LLM layer can be switched to OpenAI without
    changing retrieval code.
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def embed_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if int.from_bytes(digest[4:], "little") % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(float(np.dot(vector, vector)))
        if norm > 0:
            vector /= norm
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.vstack([self.embed_text(text) for text in texts]).astype(np.float32)

