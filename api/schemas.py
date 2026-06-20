from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class SourceResponse(BaseModel):
    source: str
    chunk_id: str
    score: float

    page_number: int | None = None
    start_char: int
    end_char: int

    text: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    include_context: bool = False


class QueryTimingsResponse(BaseModel):
    query_embedding_ms: float
    retrieval_ms: float
    prompt_build_ms: float
    generation_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    timings: QueryTimingsResponse

    generation_metadata: dict[str, object] | None = None


class RetrieveRequest(BaseModel):
    query: str = Field(
        min_length=1,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    include_context: bool = True


class RetrieveTimingsResponse(BaseModel):
    query_embedding_ms: float
    retrieval_ms: float
    rerank_ms: float
    total_ms: float


class RetrieveResponse(BaseModel):
    results: list[SourceResponse]
    timings: RetrieveTimingsResponse


class DocumentResponse(BaseModel):
    id: str
    file_name: str
    source: str
    sha256: str
    size_bytes: int
    document_count: int
    chunk_count: int
    created_at: str


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class UploadDocumentResponse(BaseModel):
    document: DocumentResponse


RetrieverKind = Literal[
    "dense",
    "hybrid",
]

RerankerKind = Literal[
    "none",
    "cross_encoder",
]

RelevancePolicyKind = Literal[
    "disabled",
    "threshold",
]


class RuntimeRetrievalConfigRequest(BaseModel):
    retriever: RetrieverKind
    reranker: RerankerKind
    relevance_policy: RelevancePolicyKind

    dense_candidate_k: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    sparse_candidate_k: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    rerank_candidate_k: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    rrf_k: int = Field(
        default=60,
        ge=1,
        le=1000,
    )

    min_relevance_score: float = Field(
        default=0.60,
    )

    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        min_length=1,
    )

    reranker_device: str = Field(
        default="cpu",
        min_length=1,
    )

    reranker_batch_size: int = Field(
        default=8,
        ge=1,
        le=128,
    )

    reranker_max_length: int = Field(
        default=512,
        ge=32,
        le=8192,
    )


class RuntimeRetrievalConfigResponse(BaseModel):
    retriever: RetrieverKind
    reranker: RerankerKind
    relevance_policy: RelevancePolicyKind
    score_kind: str

    dense_candidate_k: int
    sparse_candidate_k: int
    rerank_candidate_k: int
    rrf_k: int

    min_relevance_score: float

    reranker_model: str
    reranker_device: str
    reranker_batch_size: int
    reranker_max_length: int