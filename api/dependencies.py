from __future__ import annotations

from fastapi import HTTPException, Request, status

from rag.runtime import RAGRuntime


def get_runtime(
    request: Request,
) -> RAGRuntime:
    runtime = getattr(
        request.app.state,
        "runtime",
        None,
    )

    if runtime is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail="RAG runtime is not initialized",
        )

    if not isinstance(runtime, RAGRuntime):
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail="Invalid RAG runtime state",
        )

    return runtime