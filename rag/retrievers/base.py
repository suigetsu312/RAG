from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag.document import Chunk, RetrievedChunk


@dataclass(frozen=True, slots=True)
class RetrieverResult:
    retrieved_chunks: list[RetrievedChunk]
    query_embedding_ms: float
    retrieval_ms: float


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> RetrieverResult:
        ...

    def refresh(
        self,
        chunks: list[Chunk],
    ) -> None:
        ...