from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from config import Config, GenerationOptions
from rag.chunkers import (
    Chunker,
    FixedSizeChunker,
    RoutingChunker,
)
from rag.embeddings import LocalEmbeddingService
from rag.generators import LLMAnswerGenerator
from rag.indexing_service import (
    IndexingResult,
    IndexingService,
)
from rag.loaders import (
    DocumentLoader,
    MultiFormatDocumentLoader,
    PDFFileLoader,
    TextFileLoader,
)
from rag.prompts import PromptBuilder
from rag.rag_service import RAGService
from rag.vector_stores import FAISSVectorStore
from services.llm import LLMService


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RAGRuntime:
    rag_service: RAGService
    vector_store: FAISSVectorStore
    indexing_service: IndexingService
    indexing_result: IndexingResult | None
    documents_directory: Path
    index_directory: Path


def create_document_loader() -> DocumentLoader:
    return MultiFormatDocumentLoader(
        strategies=[
            TextFileLoader(),
            PDFFileLoader(),
        ]
    )


def create_chunker() -> Chunker:
    default_chunker = FixedSizeChunker(
        chunk_size=800,
        chunk_overlap=120,
    )

    pdf_chunker = FixedSizeChunker(
        chunk_size=1200,
        chunk_overlap=200,
    )

    return RoutingChunker(
        chunkers={
            "pdf": pdf_chunker,
        },
        default_chunker=default_chunker,
    )


def load_or_build_vector_store(
    indexing_service: IndexingService,
    documents_directory: str | Path,
    index_directory: str | Path,
) -> tuple[
    FAISSVectorStore,
    IndexingResult | None,
]:
    index_directory = Path(index_directory)

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
            f"chunks_exists={chunks_exist}"
        )

    if index_exists and chunks_exist:
        started_at = perf_counter()

        vector_store = FAISSVectorStore.load(
            index_directory
        )

        load_ms = (
            perf_counter() - started_at
        ) * 1000.0

        logger.info(
            "FAISS store loaded | vectors=%d | "
            "dimension=%d | latency_ms=%.2f",
            vector_store.count,
            vector_store.dimension,
            load_ms,
        )

        return vector_store, None

    logger.info(
        "FAISS store not found; building index | "
        "documents_directory=%s",
        documents_directory,
    )

    indexing_result = (
        indexing_service.build_directory(
            documents_directory
        )
    )

    vector_store = indexing_result.vector_store

    if not isinstance(
        vector_store,
        FAISSVectorStore,
    ):
        raise TypeError(
            "IndexingService did not create "
            "a FAISSVectorStore"
        )

    vector_store.save(index_directory)

    timings = indexing_result.timings

    logger.info(
        "FAISS store built and saved | "
        "documents=%d | chunks=%d | dimension=%d | "
        "document_load_ms=%.2f | chunking_ms=%.2f | "
        "embedding_ms=%.2f | vector_store_add_ms=%.2f | "
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

    return vector_store, indexing_result


def create_rag_runtime(
    config: Config,
    llm_service: LLMService,
    documents_directory: str | Path,
    index_directory: str | Path,
) -> RAGRuntime:
    if config.embedding.backend != "local":
        raise ValueError(
            "Only the local embedding backend "
            "is currently supported"
        )

    embedding_service = LocalEmbeddingService(
        model_name=config.embedding.model,
        device=config.embedding.device,
        batch_size=config.embedding.batch_size,
    )

    indexing_service = IndexingService(
        loader=create_document_loader(),
        chunker=create_chunker(),
        embedding_service=embedding_service,
        vector_store_factory=FAISSVectorStore,
    )

    vector_store, indexing_result = (
        load_or_build_vector_store(
            indexing_service=indexing_service,
            documents_directory=documents_directory,
            index_directory=index_directory,
        )
    )

    answer_generator = LLMAnswerGenerator(
        llm_service=llm_service,
        options=GenerationOptions(
            num_ctx=4096,
            num_predict=1024,
            temperature=0.2,
            top_p=1.0,
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
        indexing_result=indexing_result,
        documents_directory=Path(documents_directory),
        index_directory=Path(index_directory),
    )
