from __future__ import annotations

from threading import RLock
from time import perf_counter

import bm25s

from rag.document import Chunk, RetrievedChunk
from rag.retrievers.base import RetrieverResult


class BM25Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
    ) -> None:
        self._chunks: list[Chunk] = []
        self._retriever: bm25s.BM25 | None = None
        self._lock = RLock()

        self.refresh(chunks)

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

        retrieval_start = perf_counter()

        with self._lock:
            if (
                self._retriever is None
                or not self._chunks
            ):
                return RetrieverResult(
                    retrieved_chunks=[],
                    query_embedding_ms=0.0,
                    retrieval_ms=self._elapsed_ms(
                        retrieval_start
                    ),
                )

            actual_top_k = min(
                top_k,
                len(self._chunks),
            )

            query_tokens = bm25s.tokenize(
                normalized_query
            )

            indices, scores = self._retriever.retrieve(
                query_tokens,
                k=actual_top_k,
            )

            retrieved_chunks = [
                RetrievedChunk(
                    chunk=self._chunks[int(index)],
                    score=float(score),
                )
                for index, score in zip(
                    indices[0],
                    scores[0],
                    strict=True,
                )
            ]

        return RetrieverResult(
            retrieved_chunks=retrieved_chunks,
            query_embedding_ms=0.0,
            retrieval_ms=self._elapsed_ms(
                retrieval_start
            ),
        )

    def refresh(
        self,
        chunks: list[Chunk],
    ) -> None:
        with self._lock:
            self._chunks = list(chunks)

            if not self._chunks:
                self._retriever = None
                return

            corpus = [
                chunk.text
                for chunk in self._chunks
            ]

            corpus_tokens = bm25s.tokenize(
                corpus
            )

            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)

            self._retriever = retriever

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (perf_counter() - started_at) * 1000.0