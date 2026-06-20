from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from rag.document import Chunk, RetrievedChunk
from rag.metrics import RetrievalTimings
from rag.rerankers import Reranker
from rag.retrievers import Retriever


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    retrieved_chunks: list[RetrievedChunk]
    timings: RetrievalTimings


class RetrievalPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0"
            )

        self._retriever = retriever
        self._reranker = reranker
        self._candidate_k = candidate_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        total_start = perf_counter()

        candidate_k = max(
            top_k,
            self._candidate_k,
        )

        retriever_result = self._retriever.retrieve(
            query=normalized_query,
            top_k=candidate_k,
        )

        rerank_start = perf_counter()

        retrieved_chunks = self._reranker.rerank(
            query=normalized_query,
            candidates=(
                retriever_result.retrieved_chunks
            ),
            top_k=top_k,
        )

        rerank_ms = self._elapsed_ms(
            rerank_start
        )

        return RetrievalResult(
            retrieved_chunks=retrieved_chunks,
            timings=RetrievalTimings(
                query_embedding_ms=(
                    retriever_result.query_embedding_ms
                ),
                retrieval_ms=(
                    retriever_result.retrieval_ms
                ),
                rerank_ms=rerank_ms,
                total_ms=self._elapsed_ms(
                    total_start
                ),
            ),
        )

    def refresh(
        self,
        chunks: list[Chunk],
    ) -> None:
        self._retriever.refresh(chunks)

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (perf_counter() - started_at) * 1000.0