from __future__ import annotations

from rag.document import RetrievedChunk


class NoOpReranker:
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        return candidates[:top_k]