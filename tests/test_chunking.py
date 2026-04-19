from __future__ import annotations

from pathlib import Path

from app.rag.pdf_parser import DocumentPage
from app.rag.text_chunker import TextChunker


def test_chunker_creates_metadata() -> None:
    text = (
        "DHCP istemcilere otomatik IP adresi verir.\n\n"
        "DORA süreci Discover, Offer, Request ve Acknowledge adımlarından oluşur.\n\n"
        "Sunucu ayrıca varsayılan ağ geçidi ve DNS bilgisini sağlayabilir."
    )
    chunker = TextChunker(max_chars=90, overlap_chars=20)
    chunks = chunker.chunk_pages(
        [DocumentPage(text=text, page_number=2)],
        source_path=Path("dhcp_notes.md"),
        course="Computer Networks",
        topic="DHCP",
    )

    assert chunks
    assert all(chunk.source_file == "dhcp_notes.md" for chunk in chunks)
    assert all(chunk.course == "Computer Networks" for chunk in chunks)
    assert chunks[0].page_number == 2
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_id.startswith("chunk_")

