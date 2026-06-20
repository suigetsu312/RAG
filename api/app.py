from __future__ import annotations

import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
)

from pydantic import BaseModel, Field

from config import load_env
from rag.document_manifest import (
    DocumentRecord,
    calculate_sha256,
)
from rag.runtime import RAGRuntime, create_rag_runtime
from services.llm import create_llm_service


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1)
    include_context: bool = False


class SourceResponse(BaseModel):
    source: str
    chunk_id: str
    score: float
    page_number: int | None = None
    start_char: int
    end_char: int
    text: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    timings: dict[str, float]
    generation: dict[str, object]


class UploadDocumentResponse(BaseModel):
    document_id: str
    file_name: str
    sha256: str
    size_bytes: int
    document_count: int
    chunk_count: int
    vector_count: int
    timings: dict[str, float]


class DocumentResponse(BaseModel):
    id: str
    file_name: str
    source: str
    sha256: str
    size_bytes: int
    document_count: int
    chunk_count: int
    created_at: str


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentResponse]


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class RetrieveTimingsResponse(BaseModel):
    query_embedding_ms: float
    retrieval_ms: float
    total_ms: float


class RetrieveResponse(BaseModel):
    results: list[SourceResponse]
    timings: RetrieveTimingsResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_env()

    # 沿用你目前已經測通的 LLM 建立方式。
    llm_service = create_llm_service(config)

    runtime = create_rag_runtime(
        config,
        llm_service=llm_service,
        documents_directory=os.getenv(
            "RAG_DOCUMENTS_DIR",
            "data/documents",
        ),
        index_directory=os.getenv(
            "RAG_INDEX_DIR",
            "data/index",
        ),
    )

    app.state.rag_runtime = runtime

    yield


app = FastAPI(
    title="RAG Service",
    lifespan=lifespan,
)


def get_runtime(request: Request) -> RAGRuntime:
    return request.app.state.rag_runtime


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    runtime = get_runtime(request)

    return {
        "status": "ok",
        "vector_count": runtime.vector_store.count,
        "embedding_dimension": (
            runtime.vector_store.dimension
        ),
    }


@app.post(
    "/query",
    response_model=QueryResponse,
    response_model_exclude_none=True,
)
def query(
    payload: QueryRequest,
    request: Request,
) -> QueryResponse:
    runtime = get_runtime(request)

    result = runtime.rag_service.ask(
        question=payload.question,
        top_k=payload.top_k,
    )

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                source=item.chunk.source,
                chunk_id=item.chunk.id,
                score=item.score,
                page_number=item.chunk.metadata.get("page_number"),
                start_char=item.chunk.start_char,
                end_char=item.chunk.end_char,
                text=(
                    item.chunk.text
                    if payload.include_context
                    else None
                ),
            )
            for item in result.retrieved_chunks
        ],
        timings={
            "query_embedding_ms": (
                result.timings.query_embedding_ms
            ),
            "retrieval_ms": (
                result.timings.retrieval_ms
            ),
            "prompt_build_ms": (
                result.timings.prompt_build_ms
            ),
            "generation_ms": (
                result.timings.generation_ms
            ),
            "total_ms": (
                result.timings.total_ms
            ),
        },
        generation=result.generation_metadata,
    )


@app.post(
    "/documents",
    response_model=UploadDocumentResponse,
    status_code=201,
)
def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> UploadDocumentResponse:
    runtime = get_runtime(request)

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename",
        )

    file_name = Path(file.filename).name
    suffix = Path(file_name).suffix.lower()

    supported_suffixes = {
        ".txt",
        ".md",
        ".markdown",
        ".pdf",
    }

    if suffix not in supported_suffixes:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {suffix}",
        )

    destination = (
        runtime.documents_directory / file_name
    )

    temporary_path = destination.with_suffix(
        f"{destination.suffix}.uploading"
    )

    runtime.documents_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with temporary_path.open("wb") as output:
            shutil.copyfileobj(
                file.file,
                output,
            )

        sha256 = calculate_sha256(temporary_path)
        size_bytes = temporary_path.stat().st_size

        with runtime.indexing_lock:
            existing_content = (
                runtime.document_manifest.find_by_sha256(
                    sha256
                )
            )

            if existing_content is not None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": (
                            "Document with the same content "
                            "already exists"
                        ),
                        "document_id": existing_content.id,
                        "file_name": existing_content.file_name,
                    },
                )

            existing_name = (
                runtime.document_manifest.find_by_file_name(
                    file_name
                )
            )

            if existing_name is not None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": (
                            "Document with the same filename "
                            "already exists"
                        ),
                        "document_id": existing_name.id,
                    },
                )

            if destination.exists():
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"File already exists on disk: "
                        f"{file_name}"
                    ),
                )

            os.replace(
                temporary_path,
                destination,
            )

            result = runtime.indexing_service.add_file(
                path=destination,
                vector_store=runtime.vector_store,
            )

            runtime.vector_store.save(
                runtime.index_directory
            )

            record = DocumentRecord(
                id=sha256,
                file_name=file_name,
                source=file_name,
                sha256=sha256,
                size_bytes=size_bytes,
                document_count=result.document_count,
                chunk_count=result.chunk_count,
                created_at=datetime.now(
                    timezone.utc
                ).isoformat(),
            )

            runtime.document_manifest.add(record)

    except HTTPException:
        temporary_path.unlink(missing_ok=True)
        raise

    except Exception:
        temporary_path.unlink(missing_ok=True)

        if destination.exists():
            destination.unlink(missing_ok=True)

        raise

    timings = result.timings

    return UploadDocumentResponse(
        document_id=record.id,
        file_name=record.file_name,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        document_count=record.document_count,
        chunk_count=record.chunk_count,
        vector_count=runtime.vector_store.count,
        timings={
            "document_load_ms": timings.document_load_ms,
            "chunking_ms": timings.chunking_ms,
            "embedding_ms": timings.embedding_ms,
            "vector_store_add_ms": (
                timings.vector_store_add_ms
            ),
            "total_ms": timings.total_ms,
        },
    )


@app.get(
    "/documents",
    response_model=DocumentListResponse,
)
def list_documents(
    request: Request,
) -> DocumentListResponse:
    runtime = get_runtime(request)

    records = (
        runtime.document_manifest.list_documents()
    )

    return DocumentListResponse(
        total=len(records),
        documents=[
            DocumentResponse(
                id=record.id,
                file_name=record.file_name,
                source=record.source,
                sha256=record.sha256,
                size_bytes=record.size_bytes,
                document_count=record.document_count,
                chunk_count=record.chunk_count,
                created_at=record.created_at,
            )
            for record in records
        ],
    )


@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    response_model_exclude_none=True,
)
def retrieve(
    payload: RetrieveRequest,
    request: Request,
) -> RetrieveResponse:
    runtime = get_runtime(request)

    result = runtime.rag_service.retrieve(
        query=payload.query,
        top_k=payload.top_k,
    )

    return RetrieveResponse(
        results=[
            SourceResponse(
                source=item.chunk.source,
                chunk_id=item.chunk.id,
                score=item.score,
                page_number=_get_page_number(
                    item.chunk.metadata
                ),
                start_char=item.chunk.start_char,
                end_char=item.chunk.end_char,
                text=item.chunk.text,
            )
            for item in result.retrieved_chunks
        ],
        timings=RetrieveTimingsResponse(
            query_embedding_ms=(
                result.timings.query_embedding_ms
            ),
            retrieval_ms=result.timings.retrieval_ms,
            total_ms=result.timings.total_ms,
        ),
    )


def _get_page_number(
    metadata: dict[str, object],
) -> int | None:
    page_number = metadata.get("page_number")

    if isinstance(page_number, int):
        return page_number

    return None
