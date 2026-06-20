from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from api.routers.documents import (
    router as documents_router,
)
from api.routers.query import (
    router as query_router,
)
from api.routers.runtime import (
    router as runtime_router,
)
from api.schemas import HealthResponse
from config import Config, load_env
from rag.runtime import (
    RAGRuntime,
    create_rag_runtime,
)
from services.llm import (
    LLMService,
    create_llm_service,
)


logger = logging.getLogger(__name__)


DEFAULT_DOCUMENTS_DIRECTORY = Path(
    "data/documents"
)

DEFAULT_INDEX_DIRECTORY = Path(
    "data/index"
)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    logger.info(
        "Starting RAG API"
    )

    config = load_env()

    documents_directory = _get_path_from_env(
        name="RAG_DOCUMENTS_DIRECTORY",
        default=DEFAULT_DOCUMENTS_DIRECTORY,
    )

    index_directory = _get_path_from_env(
        name="RAG_INDEX_DIRECTORY",
        default=DEFAULT_INDEX_DIRECTORY,
    )

    llm_service = create_llm_service(
        config
    )

    runtime = create_rag_runtime(
        config=config,
        llm_service=llm_service,
        documents_directory=documents_directory,
        index_directory=index_directory,
    )

    app.state.config = config
    app.state.llm_service = llm_service
    app.state.runtime = runtime

    logger.info(
        "RAG API started | "
        "documents_directory=%s | "
        "index_directory=%s | "
        "vectors=%d",
        documents_directory,
        index_directory,
        runtime.vector_store.count,
    )

    try:
        yield
    finally:
        logger.info(
            "Stopping RAG API"
        )

        _close_resource(
            app.state.llm_service
        )

        app.state.runtime = None
        app.state.llm_service = None
        app.state.config = None

        logger.info(
            "RAG API stopped"
        )


def create_app() -> FastAPI:
    application = FastAPI(
        title="RAG Service",
        description=(
            "Local document indexing, retrieval, "
            "reranking, and grounded generation service."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    application.include_router(
        query_router
    )

    application.include_router(
        documents_router
    )

    application.include_router(
        runtime_router
    )

    application.add_api_route(
        path="/health",
        endpoint=health,
        methods=["GET"],
        response_model=HealthResponse,
        tags=["health"],
    )

    return application


def health() -> HealthResponse:
    return HealthResponse(
        status="ok"
    )


def _get_path_from_env(
    *,
    name: str,
    default: Path,
) -> Path:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized_value = raw_value.strip()

    if not normalized_value:
        raise ValueError(
            f"{name} must not be empty"
        )

    return Path(normalized_value)


def _close_resource(
    resource: Any,
) -> None:
    close = getattr(
        resource,
        "close",
        None,
    )

    if not callable(close):
        return

    try:
        close()
    except Exception:
        logger.exception(
            "Failed to close resource | "
            "type=%s",
            type(resource).__name__,
        )


app = create_app()