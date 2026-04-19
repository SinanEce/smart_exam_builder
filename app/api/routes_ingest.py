from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.datastructures import UploadFile

from app.core.config import Settings
from app.core.container import get_ingestion_service, settings_dependency
from app.models.schemas import IngestionResponse
from app.rag.loaders import SUPPORTED_EXTENSIONS
from app.services.ingestion_service import IngestionService

router = APIRouter(tags=["Materials"])

INGEST_MATERIALS_OPENAPI_EXTRA = {
    "requestBody": {
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "course": {
                            "type": "string",
                            "title": "Course",
                            "default": "Computer Networks",
                            "description": "Course name",
                        },
                        "folder_path": {
                            "type": "string",
                            "title": "Folder Path",
                            "description": "Optional local folder path to ingest",
                        },
                        "files": {
                            "type": "array",
                            "title": "Files",
                            "description": "Optional .txt, .md, or .pdf files",
                            "items": {"type": "string", "format": "binary"},
                        },
                    },
                },
                "encoding": {"files": {"style": "form", "explode": True}},
            }
        }
    }
}


def _is_blank_file_form_value(value: str) -> bool:
    return value.strip().lower() in {"", "null", "none", "undefined"}


def _optional_form_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, UploadFile):
        return None
    text = str(value).strip()
    return None if _is_blank_file_form_value(text) else text


def _normalize_upload_files(files: list[Any]) -> list[UploadFile]:
    if not files:
        return []

    uploads: list[UploadFile] = []
    for item in files:
        if isinstance(item, str):
            if _is_blank_file_form_value(item):
                continue
            raise HTTPException(status_code=400, detail="Invalid files field. Upload a real file or leave it blank.")
        if not isinstance(item, UploadFile):
            raise HTTPException(status_code=400, detail="Invalid files field. Upload a real file or leave it blank.")
        if not item.filename:
            continue
        uploads.append(item)
    return uploads


@router.post(
    "/ingest-materials",
    response_model=IngestionResponse,
    openapi_extra=INGEST_MATERIALS_OPENAPI_EXTRA,
)
async def ingest_materials(
    request: Request,
    service: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(settings_dependency),
) -> IngestionResponse:
    try:
        form = await request.form()
        course = _optional_form_text(form.get("course")) or "Computer Networks"
        folder_path = _optional_form_text(form.get("folder_path"))
        uploads = _normalize_upload_files(form.getlist("files"))

        saved_files: list[Path] = []
        if uploads:
            assert settings.raw_data_dir is not None
            upload_dir = settings.raw_data_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            for upload in uploads:
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
