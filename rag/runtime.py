from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from config import GenerationOptions
from rag.chunkers import FixedSizeChunker
from rag.embeddings import LocalEmbeddingService
from rag.generators import LLMAnswerGenerator
from rag.indexing_service import IndexingService
from rag.loaders import TextDocumentLoader
from rag.prompts import PromptBuilder
from rag.rag_service import RAGService
from rag.vector_stores import FAISSVectorStore
from services.llm import LLMService


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RAGRuntime:
    rag_service: RAGService
    vector_store: FAISSVectorStore
    indexing_service: IndexingService
    documents_directory: Path
    index_directory: Path


def load_or_build_vector_store(
    indexing_service: IndexingService,
    documents_directory: Path,
    index_directory: Path,
) -> FAISSVectorStore:
    index_path = (
        index_directory
        / FAISSVectorStore.INDEX_FILE_NAME
    )
    chunks_path = (
        index_directory
        / FAISSVectorStore.CHUNKS_FILE_NAME
    )

    index_exists = index_path.is_file()
    chunks_exist = chunks_path.is_file()

    if index_exists != chunks_exist:
        raise RuntimeError(
            "FAISS persistence is incomplete: "
            f"index_exists={index_exists}, "
            f"chunks_exist={chunks_exist}"
        )

    if index_exists and chunks_exist:
        started_at = perf_counter()

        vector_store = FAISSVectorStore.load(
            index_directory
        )

        elapsed_ms = (
            perf_counter() - started_at
        ) * 1000.0

        logger.info(
            "FAISS store loaded | vectors=%d | "
            "dimension=%d | latency_ms=%.2f",
            vector_store.count,
            vector_store.dimension,
            elapsed_ms,
        )

        return vector_store

    logger.info(
        "FAISS persistence not found; building index | "
        "documents_directory=%s",
        documents_directory,
    )

    indexing_result = indexing_service.build(
        documents_directory
    )

    vector_store = indexing_result.vector_store

    if not isinstance(
        vector_store,
        FAISSVectorStore,
    ):
        raise TypeError(
            "IndexingService did not create a "
            "FAISSVectorStore"
        )

    vector_store.save(index_directory)

    timings = indexing_result.timings

    logger.info(
        "FAISS store built and saved | "
        "documents=%d | chunks=%d | dimension=%d | "
        "load_ms=%.2f | chunking_ms=%.2f | "
        "embedding_ms=%.2f | add_ms=%.2f | "
        "total_ms=%.2f",
        indexing_result.document_count,
        indexing_result.chunk_count,
        indexing_result.embedding_dimension,
        timings.document_load_ms,
        timings.chunking_ms,
        timings.embedding_ms,
        timings.vector_store_add_ms,
        timings.total_ms,
    )

    return vector_store


def create_rag_runtime(
    *,
    llm_service: LLMService,
    documents_directory: str | Path,
    index_directory: str | Path,
    embedding_model: str,
    embedding_device: str,
    embedding_batch_size: int,
) -> RAGRuntime:
    documents_path = Path(documents_directory)
    index_path = Path(index_directory)

    embedding_service = LocalEmbeddingService(
        model_name=embedding_model,
        device=embedding_device,
        batch_size=embedding_batch_size,
    )

    indexing_service = IndexingService(
        loader=TextDocumentLoader(),
        chunker=FixedSizeChunker(
            chunk_size=500,
            chunk_overlap=100,
        ),
        embedding_service=embedding_service,
        vector_store_factory=FAISSVectorStore,
    )

    vector_store = load_or_build_vector_store(
        indexing_service=indexing_service,
        documents_directory=documents_path,
        index_directory=index_path,
    )

    answer_generator = LLMAnswerGenerator(
        llm_service=llm_service,
        options=GenerationOptions(
            num_ctx=4096,
            num_predict=1024,
            temperature=0.2,
        ),
    )

    rag_service = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        prompt_builder=PromptBuilder(),
        answer_generator=answer_generator,
    )

    return RAGRuntime(
        rag_service=rag_service,
        vector_store=vector_store,
        indexing_service=indexing_service,
        documents_directory=documents_path,
        index_directory=index_path,
    )