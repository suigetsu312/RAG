from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from config import load_env
from rag.runtime import RAGRuntime, create_rag_runtime
from services.llm import create_llm_service


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=5, ge=1)


class SourceResponse(BaseModel):
    source: str
    chunk_id: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    timings: dict[str, float]
    generation: dict[str, object]


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_env()

    # 沿用你目前已經測通的 LLM 建立方式。
    llm_service = create_llm_service(config)

    runtime = create_rag_runtime(
        llm_service=llm_service,
        documents_directory=os.getenv(
            "RAG_DOCUMENTS_DIR",
            "data/documents",
        ),
        index_directory=os.getenv(
            "RAG_INDEX_DIR",
            "data/index",
        ),
        embedding_model=config.embedding.model,
        embedding_device=config.embedding.device,
        embedding_batch_size=config.embedding.batch_size,
    )

    app.state.rag_runtime = runtime

    yield


app = FastAPI(
    title="RAG Service",
    lifespan=lifespan,
)


def get_runtime(request: Request) -> RAGRuntime:
    return request.app.state.rag_runtime


@app.get("/health")
def health(request: Request) -> dict[str, object]:
    runtime = get_runtime(request)

    return {
        "status": "ok",
        "vector_count": runtime.vector_store.count,
        "embedding_dimension": (
            runtime.vector_store.dimension
        ),
    }


@app.post("/query", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    request: Request,
) -> QueryResponse:
    runtime = get_runtime(request)

    result = runtime.rag_service.ask(
        question=payload.question,
        top_k=payload.top_k,
    )

    return QueryResponse(
        answer=result.answer,
        sources=[
            SourceResponse(
                source=item.chunk.source,
                chunk_id=item.chunk.id,
                score=item.score,
            )
            for item in result.retrieved_chunks
        ],
        timings={
            "query_embedding_ms": (
                result.timings.query_embedding_ms
            ),
            "retrieval_ms": (
                result.timings.retrieval_ms
            ),
            "prompt_build_ms": (
                result.timings.prompt_build_ms
            ),
            "generation_ms": (
                result.timings.generation_ms
            ),
            "total_ms": (
                result.timings.total_ms
            ),
        },
        generation=result.generation_metadata,
    )