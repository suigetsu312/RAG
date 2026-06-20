from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndexingTimings:
    document_load_ms: float
    chunking_ms: float
    embedding_ms: float
    vector_store_add_ms: float
    total_ms: float


@dataclass(frozen=True, slots=True)
class RAGTimings:
    query_embedding_ms: float = 0.0
    retrieval_ms: float = 0.0
    prompt_build_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
