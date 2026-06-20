from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import RLock
from typing import Literal

from config import RetrievalConfig
from rag.document import Chunk
from rag.embeddings import EmbeddingService
from rag.relevance import (
    DisabledRelevancePolicy,
    RelevancePolicy,
    ThresholdRelevancePolicy,
)
from rag.rerankers import (
    CrossEncoderReranker,
    NoOpReranker,
    Reranker,
)
from rag.retrieval_pipeline import RetrievalPipeline
from rag.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retriever,
)
from rag.vector_stores import FAISSVectorStore


logger = logging.getLogger(__name__)


ScoreKind = Literal[
    "cosine_similarity",
    "rrf",
    "reranker_probability",
]


@dataclass(frozen=True, slots=True)
class RetrievalComponents:
    config: RetrievalConfig
    pipeline: RetrievalPipeline
    relevance_policy: RelevancePolicy
    score_kind: ScoreKind


class RetrievalComponentManager:
    def __init__(
        self,
        initial_components: RetrievalComponents,
    ) -> None:
        self._components = initial_components
        self._lock = RLock()

    def snapshot(self) -> RetrievalComponents:
        with self._lock:
            return self._components

    def replace(
        self,
        components: RetrievalComponents,
    ) -> None:
        with self._lock:
            self._components = components

    def refresh(
        self,
        chunks: list[Chunk],
    ) -> None:
        components = self.snapshot()
        components.pipeline.refresh(chunks)


def create_retrieval_components(
    *,
    config: RetrievalConfig,
    embedding_service: EmbeddingService,
    vector_store: FAISSVectorStore,
) -> RetrievalComponents:
    retriever = _create_retriever(
        config=config,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    reranker = _create_reranker(config)

    relevance_policy = (
        _create_relevance_policy(config)
    )

    pipeline = RetrievalPipeline(
        retriever=retriever,
        reranker=reranker,
        candidate_k=config.rerank_candidate_k,
    )

    components = RetrievalComponents(
        config=config,
        pipeline=pipeline,
        relevance_policy=relevance_policy,
        score_kind=_determine_score_kind(config),
    )

    logger.info(
        "Retrieval components created | "
        "retriever=%s | reranker=%s | "
        "relevance_policy=%s | score_kind=%s",
        config.retriever,
        config.reranker,
        config.relevance_policy,
        components.score_kind,
    )

    return components


def _create_retriever(
    *,
    config: RetrievalConfig,
    embedding_service: EmbeddingService,
    vector_store: FAISSVectorStore,
) -> Retriever:
    dense_retriever = DenseRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    if config.retriever == "dense":
        logger.info(
            "Dense retriever selected | "
            "candidate_k=%d",
            config.dense_candidate_k,
        )

        return dense_retriever

    if config.retriever == "hybrid":
        logger.info(
            "Hybrid retriever selected | "
            "dense_candidate_k=%d | "
            "sparse_candidate_k=%d | rrf_k=%d",
            config.dense_candidate_k,
            config.sparse_candidate_k,
            config.rrf_k,
        )

        sparse_retriever = BM25Retriever(
            chunks=list(vector_store.chunks)
        )

        return HybridRetriever(
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            dense_candidate_k=(
                config.dense_candidate_k
            ),
            sparse_candidate_k=(
                config.sparse_candidate_k
            ),
            rrf_k=config.rrf_k,
        )

    raise ValueError(
        f"Unsupported retriever: {config.retriever}"
    )


def _create_reranker(
    config: RetrievalConfig,
) -> Reranker:
    if config.reranker == "none":
        logger.info(
            "No-op reranker selected"
        )

        return NoOpReranker()

    if config.reranker == "cross_encoder":
        logger.info(
            "Cross-encoder reranker selected | "
            "model=%s | device=%s | "
            "batch_size=%d | max_length=%d",
            config.reranker_model,
            config.reranker_device,
            config.reranker_batch_size,
            config.reranker_max_length,
        )

        return CrossEncoderReranker(
            model_name=config.reranker_model,
            device=config.reranker_device,
            batch_size=config.reranker_batch_size,
            max_length=config.reranker_max_length,
        )

    raise ValueError(
        f"Unsupported reranker: {config.reranker}"
    )


def _create_relevance_policy(
    config: RetrievalConfig,
) -> RelevancePolicy:
    if config.relevance_policy == "disabled":
        logger.info(
            "Disabled relevance policy selected"
        )

        return DisabledRelevancePolicy()

    if config.relevance_policy == "threshold":
        if (
            config.retriever == "hybrid"
            and config.reranker == "none"
        ):
            logger.warning(
                "Threshold policy is using RRF scores; "
                "do not reuse a cosine-similarity threshold"
            )

        logger.info(
            "Threshold relevance policy selected | "
            "min_score=%.4f",
            config.min_relevance_score,
        )

        return ThresholdRelevancePolicy(
            min_score=config.min_relevance_score
        )

    raise ValueError(
        "Unsupported relevance policy: "
        f"{config.relevance_policy}"
    )


def _determine_score_kind(
    config: RetrievalConfig,
) -> ScoreKind:
    if config.reranker == "cross_encoder":
        return "reranker_probability"

    if config.retriever == "hybrid":
        return "rrf"

    return "cosine_similarity"