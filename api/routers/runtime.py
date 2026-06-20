from __future__ import annotations

import logging

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from api.dependencies import get_runtime
from api.schemas import (
    RuntimeRetrievalConfigRequest,
    RuntimeRetrievalConfigResponse,
)
from config import RetrievalConfig
from rag.runtime import (
    RAGRuntime,
    reconfigure_retrieval,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/runtime",
    tags=["runtime"],
)


@router.get(
    "/retrieval",
    response_model=RuntimeRetrievalConfigResponse,
)
def get_retrieval_config(
    runtime: RAGRuntime = Depends(get_runtime),
) -> RuntimeRetrievalConfigResponse:
    return _build_response(runtime)


@router.put(
    "/retrieval",
    response_model=RuntimeRetrievalConfigResponse,
)
def update_retrieval_config(
    payload: RuntimeRetrievalConfigRequest,
    runtime: RAGRuntime = Depends(get_runtime),
) -> RuntimeRetrievalConfigResponse:
    config = RetrievalConfig(
        retriever=payload.retriever,
        reranker=payload.reranker,
        relevance_policy=payload.relevance_policy,
        dense_candidate_k=payload.dense_candidate_k,
        sparse_candidate_k=payload.sparse_candidate_k,
        rerank_candidate_k=payload.rerank_candidate_k,
        rrf_k=payload.rrf_k,
        min_relevance_score=payload.min_relevance_score,
        reranker_model=payload.reranker_model,
        reranker_device=payload.reranker_device,
        reranker_batch_size=payload.reranker_batch_size,
        reranker_max_length=payload.reranker_max_length,
    )

    logger.info(
        "Runtime retrieval update requested | "
        "retriever=%s | reranker=%s | "
        "relevance_policy=%s",
        config.retriever,
        config.reranker,
        config.relevance_policy,
    )

    try:
        components = reconfigure_retrieval(
            runtime=runtime,
            config=config,
        )
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(error),
        ) from error
    except (OSError, RuntimeError) as error:
        logger.exception(
            "Failed to initialize retrieval components"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to initialize retrieval components: "
                f"{error}"
            ),
        ) from error
    except Exception as error:
        logger.exception(
            "Unexpected retrieval reconfiguration failure"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to reconfigure retrieval"
            ),
        ) from error

    logger.info(
        "Runtime retrieval update completed | "
        "retriever=%s | reranker=%s | "
        "score_kind=%s",
        components.config.retriever,
        components.config.reranker,
        components.score_kind,
    )

    return _build_response(runtime)


def _build_response(
    runtime: RAGRuntime,
) -> RuntimeRetrievalConfigResponse:
    components = (
        runtime.retrieval_components.snapshot()
    )

    config = components.config

    return RuntimeRetrievalConfigResponse(
        retriever=config.retriever,
        reranker=config.reranker,
        relevance_policy=config.relevance_policy,
        score_kind=components.score_kind,
        dense_candidate_k=config.dense_candidate_k,
        sparse_candidate_k=config.sparse_candidate_k,
        rerank_candidate_k=config.rerank_candidate_k,
        rrf_k=config.rrf_k,
        min_relevance_score=(
            config.min_relevance_score
        ),
        reranker_model=config.reranker_model,
        reranker_device=config.reranker_device,
        reranker_batch_size=(
            config.reranker_batch_size
        ),
        reranker_max_length=(
            config.reranker_max_length
        ),
    )