from __future__ import annotations

from app.models.schemas import TextChunk
from app.rag.embeddings import HashingEmbeddingModel
from app.rag.vector_store import VectorStore
from app.services.rag_service import RagService


def test_retrieval_returns_relevant_chunk(workspace_tmp) -> None:
    embedding_model = HashingEmbeddingModel(dimension=128)
    vector_store = VectorStore(storage_dir=workspace_tmp, dimension=128)
    chunks = [
        TextChunk(
            chunk_id="chunk_dhcp",
            text="DHCP DORA süreci Discover Offer Request Acknowledge adımlarından oluşur.",
            source_file="dhcp.txt",
            course="Computer Networks",
            topic="DHCP",
            chunk_index=0,
        ),
        TextChunk(
            chunk_id="chunk_nat",
            text="NAT özel IP adreslerini genel IP adresiyle internete çıkarır.",
            source_file="nat.txt",
            course="Computer Networks",
            topic="NAT",
            chunk_index=0,
        ),
    ]
    vector_store.upsert(chunks, embedding_model.embed_documents([chunk.text for chunk in chunks]))

    service = RagService(embedding_model=embedding_model, vector_store=vector_store)
    results = service.search("DHCP Discover Offer Request Acknowledge", course="Computer Networks", top_k=1)

    assert results
    assert results[0].chunk_id == "chunk_dhcp"
    assert results[0].preview_text
