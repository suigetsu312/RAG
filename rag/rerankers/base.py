from __future__ import annotations

from typing import Protocol

from rag.document import RetrievedChunk


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        ...