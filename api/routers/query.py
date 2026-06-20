from __future__ import annotations

from typing import Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from api.dependencies import get_runtime
from api.schemas import (
    QueryRequest,
    QueryResponse,
    QueryTimingsResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrieveTimingsResponse,
    SourceResponse,
)
from rag.document import RetrievedChunk
from rag.runtime import RAGRuntime


router = APIRouter(
    tags=["query"],
)


@router.post(
    "/query",
    response_model=QueryResponse,
    response_model_exclude_none=True,
)
def query(
    payload: QueryRequest,
    runtime: RAGRuntime = Depends(get_runtime),
) -> QueryResponse:
    try:
        result = runtime.rag_service.ask(
            question=payload.question,
            top_k=payload.top_k,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except RuntimeError as error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(error),
        ) from error

    return QueryResponse(
        answer=result.answer,
        sources=[
            _build_source_response(
                item,
                include_context=payload.include_context,
            )
            for item in result.retrieved_chunks
        ],
        timings=QueryTimingsResponse(
            query_embedding_ms=(
                result.timings.query_embedding_ms
            ),
            retrieval_ms=(
                result.timings.retrieval_ms
            ),
            prompt_build_ms=(
                result.timings.prompt_build_ms
            ),
            generation_ms=(
                result.timings.generation_ms
            ),
            total_ms=result.timings.total_ms,
        ),
        generation_metadata=(
            _normalize_generation_metadata(
                result.generation_metadata
            )
        ),
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    response_model_exclude_none=True,
)
def retrieve(
    payload: RetrieveRequest,
    runtime: RAGRuntime = Depends(get_runtime),
) -> RetrieveResponse:
    try:
        result = runtime.rag_service.retrieve(
            query=payload.query,
            top_k=payload.top_k,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error

    return RetrieveResponse(
        results=[
            _build_source_response(
                item,
                include_context=payload.include_context,
            )
            for item in result.retrieved_chunks
        ],
        timings=RetrieveTimingsResponse(
            query_embedding_ms=(
                result.timings.query_embedding_ms
            ),
            retrieval_ms=result.timings.retrieval_ms,
            rerank_ms=result.timings.rerank_ms,
            total_ms=result.timings.total_ms,
        ),
    )


def _build_source_response(
    item: RetrievedChunk,
    *,
    include_context: bool,
) -> SourceResponse:
    return SourceResponse(
        source=item.chunk.source,
        chunk_id=item.chunk.id,
        score=item.score,
        page_number=_get_page_number(
            item.chunk.metadata
        ),
        start_char=item.chunk.start_char,
        end_char=item.chunk.end_char,
        text=(
            item.chunk.text
            if include_context
            else None
        ),
    )


def _get_page_number(
    metadata: dict[str, object],
) -> int | None:
    page_number = metadata.get("page_number")

    if isinstance(page_number, int):
        return page_number

    return None


def _normalize_generation_metadata(
    metadata: object | None,
) -> dict[str, object] | None:
    if metadata is None:
        return None

    if isinstance(metadata, dict):
        return {
            str(key): value
            for key, value in metadata.items()
        }

    if hasattr(metadata, "__dict__"):
        values = vars(metadata)

        return {
            str(key): value
            for key, value in values.items()
        }

    return {
        "value": metadata,
    }