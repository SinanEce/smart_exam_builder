from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.config import Settings
from app.core.container import get_ingestion_service, settings_dependency
from app.models.schemas import IngestionResponse
from app.rag.loaders import SUPPORTED_EXTENSIONS
from app.services.ingestion_service import IngestionService

router = APIRouter(tags=["Materials"])


@router.post("/ingest-materials", response_model=IngestionResponse)
async def ingest_materials(
    course: Annotated[str, Form(description="Course name")] = "Computer Networks",
    folder_path: Annotated[str | None, Form(description="Optional local folder path to ingest")] = None,
    files: Annotated[list[UploadFile] | None, File(description="Optional .txt, .md, or .pdf files")] = None,
    service: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(settings_dependency),
) -> IngestionResponse:
    try:
        saved_files: list[Path] = []
        if files:
            assert settings.raw_data_dir is not None
            upload_dir = settings.raw_data_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            for upload in files:
                filename = Path(upload.filename or "uploaded_material.txt").name
                suffix = Path(filename).suffix.lower()
                if suffix not in SUPPORTED_EXTENSIONS:
                    raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
                target = upload_dir / filename
                target.write_bytes(await upload.read())
                saved_files.append(target)

        if folder_path:
            folder = Path(folder_path).expanduser().resolve()
            response = service.ingest_local_folder(folder, course=course)
            if saved_files:
                upload_response = service.ingest_files(saved_files, course=course)
                response.indexed_chunks += upload_response.indexed_chunks
                response.total_chunks_in_store = upload_response.total_chunks_in_store
                response.source_files.extend(upload_response.source_files)
                response.chunks.extend(upload_response.chunks)
            return response

        if saved_files:
            return service.ingest_files(saved_files, course=course)

        assert settings.raw_data_dir is not None
        return service.ingest_local_folder(settings.raw_data_dir, course=course)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

