from __future__ import annotations

import re
from pathlib import Path

from app.models.schemas import TextChunk
from app.rag.pdf_parser import DocumentPage
from app.utils.ids import stable_id


class TextChunker:
    def __init__(self, max_chars: int = 900, overlap_chars: int = 160) -> None:
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars.")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk_pages(
        self,
        pages: list[DocumentPage],
        source_path: Path,
        course: str,
        topic: str | None,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_index = 0
        for page in pages:
            for text in self._chunk_text(page.text):
                chunk_id = stable_id(
                    "chunk",
                    source_path.name,
                    course,
                    topic or "",
                    page.page_number or "",
                    chunk_index,
                    text[:80],
                )
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=text,
                        source_file=source_path.name,
                        course=course,
                        topic=topic,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
        return chunks

    def _chunk_text(self, text: str) -> list[str]:
        cleaned = self._normalize(text)
        if not cleaned:
            return []

        paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", cleaned) if paragraph.strip()]
        chunks: list[str] = []
        current = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.max_chars:
                if current:
                    chunks.append(current.strip())
                    current = ""
                chunks.extend(self._split_long_text(paragraph))
                continue

            candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
            if len(candidate) <= self.max_chars:
                current = candidate
            else:
                chunks.append(current.strip())
                current = self._with_overlap(current, paragraph)

        if current:
            chunks.append(current.strip())

        return [chunk for chunk in chunks if chunk.strip()]

    def _split_long_text(self, text: str) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        current_words: list[str] = []
        current_len = 0

        for word in words:
            addition = len(word) + (1 if current_words else 0)
            if current_words and current_len + addition > self.max_chars:
                chunk = " ".join(current_words)
                chunks.append(chunk)
                overlap = self._tail_words(chunk)
                current_words = overlap + [word]
                current_len = len(" ".join(current_words))
            else:
                current_words.append(word)
                current_len += addition

        if current_words:
            chunks.append(" ".join(current_words))
        return chunks

    def _with_overlap(self, previous: str, next_paragraph: str) -> str:
        tail = previous[-self.overlap_chars :].strip()
        if tail:
            return f"{tail}\n\n{next_paragraph}".strip()
        return next_paragraph

    def _tail_words(self, text: str) -> list[str]:
        tail = text[-self.overlap_chars :].split()
        return tail[-30:]

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(lines).strip()

