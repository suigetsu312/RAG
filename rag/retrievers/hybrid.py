from __future__ import annotations

from collections import defaultdict
from time import perf_counter

from rag.document import Chunk, RetrievedChunk
from rag.retrievers.base import (
    Retriever,
    RetrieverResult,
)


class HybridRetriever:
    def __init__(
        self,
        dense_retriever: Retriever,
        sparse_retriever: Retriever,
        dense_candidate_k: int = 20,
        sparse_candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if dense_candidate_k <= 0:
            raise ValueError(
                "dense_candidate_k must be greater than 0"
            )

        if sparse_candidate_k <= 0:
            raise ValueError(
                "sparse_candidate_k must be greater than 0"
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than 0"
            )

        self._dense_retriever = dense_retriever
        self._sparse_retriever = sparse_retriever
        self._dense_candidate_k = dense_candidate_k
        self._sparse_candidate_k = sparse_candidate_k
        self._rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> RetrieverResult:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        dense_result = self._dense_retriever.retrieve(
            query=query,
            top_k=self._dense_candidate_k,
        )

        sparse_result = self._sparse_retriever.retrieve(
            query=query,
            top_k=self._sparse_candidate_k,
        )

        fusion_start = perf_counter()

        chunks_by_id: dict[str, Chunk] = {}
        fused_scores: dict[str, float] = defaultdict(float)

        self._accumulate_rrf(
            results=dense_result.retrieved_chunks,
            chunks_by_id=chunks_by_id,
            fused_scores=fused_scores,
        )

        self._accumulate_rrf(
            results=sparse_result.retrieved_chunks,
            chunks_by_id=chunks_by_id,
            fused_scores=fused_scores,
        )

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.__getitem__,
            reverse=True,
        )

        retrieved_chunks = [
            RetrievedChunk(
                chunk=chunks_by_id[chunk_id],
                score=fused_scores[chunk_id],
            )
            for chunk_id in ranked_ids[:top_k]
        ]

        fusion_ms = (
            perf_counter() - fusion_start
        ) * 1000.0

        return RetrieverResult(
            retrieved_chunks=retrieved_chunks,
            query_embedding_ms=(
                dense_result.query_embedding_ms
            ),
            retrieval_ms=(
                dense_result.retrieval_ms
                + sparse_result.retrieval_ms
                + fusion_ms
            ),
        )

    def refresh(
        self,
        chunks: list[Chunk],
    ) -> None:
        self._dense_retriever.refresh(chunks)
        self._sparse_retriever.refresh(chunks)

    def _accumulate_rrf(
        self,
        results: list[RetrievedChunk],
        chunks_by_id: dict[str, Chunk],
        fused_scores: dict[str, float],
    ) -> None:
        for rank, item in enumerate(
            results,
            start=1,
        ):
            chunk_id = item.chunk.id

            chunks_by_id[chunk_id] = item.chunk

            fused_scores[chunk_id] += (
                1.0
                / (self._rrf_k + rank)
            )