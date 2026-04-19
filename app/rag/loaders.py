from __future__ import annotations

from pathlib import Path

from app.rag.pdf_parser import DocumentPage, parse_pdf


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def infer_topic(path: Path) -> str | None:
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(part.capitalize() for part in stem.split()) if stem else None


def load_document(path: Path) -> list[DocumentPage]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported material type: {suffix}")

    if suffix == ".pdf":
        return parse_pdf(path)

    text = path.read_text(encoding="utf-8")
    return [DocumentPage(text=text, page_number=None)]


def iter_supported_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Material folder not found: {folder}")
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

