from __future__ import annotations

import logging
from pathlib import Path

from app.models.schemas import ChunkPreview, IngestionResponse, TextChunk
from app.rag.embeddings import HashingEmbeddingModel
from app.rag.loaders import infer_topic, iter_supported_files, load_document
from app.rag.text_chunker import TextChunker
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    """Loads course materials, chunks them, embeds them, and indexes them."""

    def __init__(
        self,
        chunker: TextChunker,
        embedding_model: HashingEmbeddingModel,
        vector_store: VectorStore,
    ) -> None:
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def ingest_local_folder(self, folder_path: Path, course: str) -> IngestionResponse:
        files = iter_supported_files(folder_path)
        return self.ingest_files(files, course=course)

    def ingest_files(self, files: list[Path], course: str) -> IngestionResponse:
        all_chunks: list[TextChunk] = []
        source_files: list[str] = []

        for path in files:
            try:
                pages = load_document(path)
                topic = infer_topic(path)
                chunks = self.chunker.chunk_pages(pages, source_path=path, course=course, topic=topic)
                all_chunks.extend(chunks)
                source_files.append(path.name)
                logger.info("Ingested %s into %s chunks.", path.name, len(chunks))
            except Exception:
                logger.exception("Failed to ingest %s.", path)
                raise

        if all_chunks:
            vectors = self.embedding_model.embed_documents([chunk.text for chunk in all_chunks])
            self.vector_store.upsert(all_chunks, vectors)

        return IngestionResponse(
            course=course,
            indexed_chunks=len(all_chunks),
            total_chunks_in_store=self.vector_store.count,
            source_files=source_files,
            chunks=[ChunkPreview.from_chunk(chunk) for chunk in all_chunks],
        )

