from __future__ import annotations

import os
import shutil
from contextlib import asynccontextmanager
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
from rag.runtime import RAGRuntime, create_rag_runtime
from services.llm import create_llm_service


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1)


class SourceResponse(BaseModel):
    source: str
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    timings: dict[str, float]
    generation: dict[str, object]


class UploadDocumentResponse(BaseModel):
    file_name: str
    document_count: int
    chunk_count: int
    vector_count: int
    timings: dict[str, float]


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


@app.post("/query", response_model=QueryResponse)
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

    if destination.exists():
        raise HTTPException(
            status_code=409,
            detail=f"Document already exists: {file_name}",
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

    except Exception:
        temporary_path.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise

    timings = result.timings

    return UploadDocumentResponse(
        file_name=file_name,
        document_count=result.document_count,
        chunk_count=result.chunk_count,
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
