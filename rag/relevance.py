from __future__ import annotations

from typing import Protocol

from rag.document import RetrievedChunk


class RelevancePolicy(Protocol):
    def is_relevant(
        self,
        chunks: list[RetrievedChunk],
    ) -> bool:
        ...


class DisabledRelevancePolicy:
    def is_relevant(
        self,
        chunks: list[RetrievedChunk],
    ) -> bool:
        return bool(chunks)


class ThresholdRelevancePolicy:
    def __init__(
        self,
        min_score: float,
    ) -> None:
        self._min_score = min_score

    def is_relevant(
        self,
        chunks: list[RetrievedChunk],
    ) -> bool:
        return (
            bool(chunks)
            and chunks[0].score >= self._min_score
        )