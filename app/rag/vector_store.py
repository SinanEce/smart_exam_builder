from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.models.schemas import TextChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Small FAISS-backed vector store with JSON metadata persistence."""

    def __init__(self, storage_dir: Path, dimension: int) -> None:
        self.storage_dir = storage_dir
        self.dimension = dimension
        self.metadata_path = storage_dir / "chunks.json"
        self.vectors_path = storage_dir / "vectors.npy"
        self.faiss_path = storage_dir / "index.faiss"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._chunks: list[TextChunk] = []
        self._vectors = np.empty((0, dimension), dtype=np.float32)
        self._faiss = self._try_import_faiss()
        self._index = None
        self.load()

    @staticmethod
    def _try_import_faiss():
        try:
            import faiss  # type: ignore

            return faiss
        except ImportError:
            logger.warning("faiss is not installed; using numpy similarity fallback.")
            return None

    @property
    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[TextChunk]:
        return list(self._chunks)

    def load(self) -> None:
        if self.metadata_path.exists():
            raw_chunks = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self._chunks = [TextChunk.model_validate(item) for item in raw_chunks]
        if self.vectors_path.exists():
            self._vectors = np.load(self.vectors_path).astype(np.float32)
        if len(self._chunks) != len(self._vectors):
            logger.warning("Vector metadata mismatch; clearing vector store.")
            self.clear()
            return
        self._rebuild_faiss()

    def clear(self) -> None:
        self._chunks = []
        self._vectors = np.empty((0, self.dimension), dtype=np.float32)
        self._index = None
        for path in [self.metadata_path, self.vectors_path, self.faiss_path]:
            if path.exists():
                path.unlink()

    def upsert(self, chunks: list[TextChunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        if vectors.shape != (len(chunks), self.dimension):
            raise ValueError("Vector shape does not match chunk count and configured dimension.")

        by_id: dict[str, tuple[TextChunk, np.ndarray]] = {
            chunk.chunk_id: (chunk, self._vectors[index]) for index, chunk in enumerate(self._chunks)
        }
        for chunk, vector in zip(chunks, vectors, strict=True):
            by_id[chunk.chunk_id] = (chunk, vector.astype(np.float32))

        ordered = list(by_id.values())
        self._chunks = [item[0] for item in ordered]
        self._vectors = np.vstack([item[1] for item in ordered]).astype(np.float32)
        self._persist()
        self._rebuild_faiss()

    def search(self, query_vector: np.ndarray, top_k: int, course: str | None = None) -> list[tuple[TextChunk, float]]:
        if self.count == 0:
            return []
        query = query_vector.astype(np.float32).reshape(1, self.dimension)
        candidate_count = min(self.count, max(top_k * 5, top_k))

        if self._index is not None:
            scores, indices = self._index.search(query, candidate_count)
            pairs = [
                (self._chunks[int(index)], float(score))
                for index, score in zip(indices[0], scores[0], strict=False)
                if int(index) >= 0
            ]
        else:
            scores = self._vectors @ query.reshape(self.dimension)
            sorted_indices = np.argsort(scores)[::-1][:candidate_count]
            pairs = [(self._chunks[int(index)], float(scores[int(index)])) for index in sorted_indices]

        if course:
            pairs = [pair for pair in pairs if pair[0].course.lower() == course.lower()]
        return pairs[:top_k]

    def _persist(self) -> None:
        self.metadata_path.write_text(
            json.dumps([chunk.model_dump(mode="json") for chunk in self._chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(self.vectors_path, self._vectors)
        if self._faiss is not None and self._index is not None:
            self._faiss.write_index(self._index, str(self.faiss_path))

    def _rebuild_faiss(self) -> None:
        if self._faiss is None or len(self._vectors) == 0:
            self._index = None
            return
        index = self._faiss.IndexFlatIP(self.dimension)
        index.add(self._vectors)
        self._index = index

