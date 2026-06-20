from __future__ import annotations

from time import perf_counter

from rag.document import Chunk
from rag.embeddings import EmbeddingService
from rag.retrievers.base import RetrieverResult
from rag.vector_stores import VectorStore


class DenseRetriever:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> RetrieverResult:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        embedding_start = perf_counter()

        embedding_result = self._embedding_service.embed(
            normalized_query
        )

        query_embedding_ms = self._elapsed_ms(
            embedding_start
        )

        retrieval_start = perf_counter()

        retrieved_chunks = self._vector_store.search(
            query_embedding=embedding_result.embedding,
            top_k=top_k,
        )

        retrieval_ms = self._elapsed_ms(
            retrieval_start
        )

        return RetrieverResult(
            retrieved_chunks=retrieved_chunks,
            query_embedding_ms=query_embedding_ms,
            retrieval_ms=retrieval_ms,
        )

    def refresh(
        self,
        chunks: list[Chunk],
    ) -> None:
        # Dense retrieval directly uses the vector store.
        # The vector store has already been updated by IndexingService.
        return None

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (perf_counter() - started_at) * 1000.0