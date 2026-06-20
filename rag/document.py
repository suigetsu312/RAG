from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.metrics import RAGTimings


@dataclass(frozen=True, slots=True)
class Document:
    id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    id: str
    document_id: str
    text: str
    source: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class RAGResult:
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    timings: RAGTimings = field(default_factory=RAGTimings)
    generation_metadata: dict[str, Any] = field(default_factory=dict)
