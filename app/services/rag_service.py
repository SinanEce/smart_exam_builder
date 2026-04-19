from __future__ import annotations

from app.models.schemas import RetrievedChunk
from app.rag.embeddings import HashingEmbeddingModel
from app.rag.vector_store import VectorStore


class RagService:
    """Search facade over embeddings and the vector store."""

    def __init__(self, embedding_model: HashingEmbeddingModel, vector_store: VectorStore) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def search(self, query: str, course: str | None, top_k: int) -> list[RetrievedChunk]:
        query_vector = self.embedding_model.embed_text(query)
        results = self.vector_store.search(query_vector=query_vector, top_k=top_k, course=course)
        return [RetrievedChunk.from_chunk(chunk, score=score) for chunk, score in results]

    @property
    def chunk_count(self) -> int:
        return self.vector_store.count

