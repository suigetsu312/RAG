from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from api.dependencies import get_runtime
from api.schemas import (
    DocumentListResponse,
    DocumentResponse,
    UploadDocumentResponse,
)
from rag.document_manifest import (
    DocumentRecord,
    calculate_sha256,
)
from rag.index_bootstrap import (
    SUPPORTED_DOCUMENT_SUFFIXES,
)
from rag.runtime import RAGRuntime


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


@router.get(
    "",
    response_model=DocumentListResponse,
)
def list_documents(
    runtime: RAGRuntime = Depends(get_runtime),
) -> DocumentListResponse:
    records = sorted(
        runtime.document_manifest.records,
        key=lambda record: record.created_at,
        reverse=True,
    )

    return DocumentListResponse(
        documents=[
            _build_document_response(record)
            for record in records
        ]
    )


@router.post(
    "",
    response_model=UploadDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    file: UploadFile = File(...),
    runtime: RAGRuntime = Depends(get_runtime),
) -> UploadDocumentResponse:
    file_name = _validate_file_name(
        file.filename
    )

    destination = (
        runtime.documents_directory
        / file_name
    )

    temporary_path = (
        runtime.documents_directory
        / f".upload-{uuid4().hex}.tmp"
    )

    try:
        _save_upload_to_temporary_file(
            upload=file,
            temporary_path=temporary_path,
        )

        sha256 = calculate_sha256(
            temporary_path
        )

        size_bytes = temporary_path.stat().st_size

        if size_bytes == 0:
            raise HTTPException(
                status_code=(
                    status.HTTP_400_BAD_REQUEST
                ),
                detail="Uploaded document is empty",
            )

        with runtime.indexing_lock:
            _ensure_document_is_not_duplicate(
                runtime=runtime,
                sha256=sha256,
            )

            _ensure_destination_is_available(
                destination
            )

            temporary_path.replace(
                destination
            )

            try:
                indexing_result = (
                    runtime.indexing_service.add_file(
                        path=destination,
                        vector_store=(
                            runtime.vector_store
                        ),
                    )
                )

                runtime.vector_store.save(
                    runtime.index_directory
                )

                record = DocumentRecord(
                    id=sha256,
                    file_name=file_name,
                    source=(
                        destination
                        .relative_to(
                            runtime.documents_directory
                        )
                        .as_posix()
                    ),
                    sha256=sha256,
                    size_bytes=size_bytes,
                    document_count=(
                        indexing_result.document_count
                    ),
                    chunk_count=(
                        indexing_result.chunk_count
                    ),
                    created_at=(
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                )

                existing_records = list(
                    runtime.document_manifest.records
                )

                runtime.document_manifest.replace_all(
                    [
                        *existing_records,
                        record,
                    ]
                )

                runtime.retrieval_components.refresh(
                    list(
                        runtime.vector_store.chunks
                    )
                )

            except Exception:
                logger.exception(
                    "Document indexing failed | "
                    "file=%s",
                    destination,
                )

                destination.unlink(
                    missing_ok=True
                )

                raise

    except HTTPException:
        temporary_path.unlink(
            missing_ok=True
        )

        raise

    except ValueError as error:
        temporary_path.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=str(error),
        ) from error

    except Exception as error:
        temporary_path.unlink(
            missing_ok=True
        )

        logger.exception(
            "Document upload failed | file=%s",
            file_name,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to upload and index document"
            ),
        ) from error

    logger.info(
        "Document uploaded | "
        "file=%s | size_bytes=%d | "
        "documents=%d | chunks=%d",
        record.file_name,
        record.size_bytes,
        record.document_count,
        record.chunk_count,
    )

    return UploadDocumentResponse(
        document=_build_document_response(
            record
        )
    )


def _validate_file_name(
    raw_file_name: str | None,
) -> str:
    if raw_file_name is None:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="Uploaded file has no filename",
        )

    file_name = Path(raw_file_name).name.strip()

    if not file_name:
        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail="Uploaded file has no filename",
        )

    suffix = Path(file_name).suffix.lower()

    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        supported = ", ".join(
            sorted(
                SUPPORTED_DOCUMENT_SUFFIXES
            )
        )

        raise HTTPException(
            status_code=(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            ),
            detail=(
                "Unsupported document type. "
                f"Supported suffixes: {supported}"
            ),
        )

    return file_name


def _save_upload_to_temporary_file(
    *,
    upload: UploadFile,
    temporary_path: Path,
) -> None:
    upload.file.seek(0)

    with temporary_path.open("wb") as output:
        shutil.copyfileobj(
            upload.file,
            output,
            length=1024 * 1024,
        )


def _ensure_document_is_not_duplicate(
    *,
    runtime: RAGRuntime,
    sha256: str,
) -> None:
    duplicate = next(
        (
            record
            for record
            in runtime.document_manifest.records
            if record.sha256 == sha256
        ),
        None,
    )

    if duplicate is None:
        return

    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=(
            "The same document content "
            "has already been indexed: "
            f"{duplicate.file_name}"
        ),
    )


def _ensure_destination_is_available(
    destination: Path,
) -> None:
    if not destination.exists():
        return

    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=(
            "A document with the same filename "
            "already exists"
        ),
    )


def _build_document_response(
    record: DocumentRecord,
) -> DocumentResponse:
    return DocumentResponse(
        id=record.id,
        file_name=record.file_name,
        source=record.source,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        document_count=record.document_count,
        chunk_count=record.chunk_count,
        created_at=record.created_at,
    )