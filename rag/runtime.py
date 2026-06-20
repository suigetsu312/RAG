from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from config import (
    Config,
    GenerationOptions,
    RetrievalConfig,
)
from rag.chunkers import (
    Chunker,
    FixedSizeChunker,
    RoutingChunker,
)
from rag.document_manifest import DocumentManifest
from rag.embeddings import (
    EmbeddingService,
    LocalEmbeddingService,
)
from rag.generators import LLMAnswerGenerator
from rag.index_bootstrap import bootstrap_vector_store
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
from rag.retrieval_runtime import (
    RetrievalComponentManager,
    RetrievalComponents,
    create_retrieval_components,
)
from rag.vector_stores import FAISSVectorStore
from services.llm import LLMService


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RAGRuntime:
    rag_service: RAGService
    retrieval_components: RetrievalComponentManager
    embedding_service: EmbeddingService
    vector_store: FAISSVectorStore
    indexing_service: IndexingService
    indexing_result: IndexingResult | None
    documents_directory: Path
    index_directory: Path
    document_manifest: DocumentManifest
    indexing_lock: threading.RLock


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

    documents_root = Path(documents_directory)
    index_root = Path(index_directory)

    documents_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    index_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    embedding_service = (
        _create_embedding_service(config)
    )

    indexing_service = (
        _create_indexing_service(
            embedding_service
        )
    )

    document_manifest = DocumentManifest.load(
        index_root / "documents.json"
    )

    vector_store, indexing_result = (
        bootstrap_vector_store(
            indexing_service=indexing_service,
            embedding_service=embedding_service,
            documents_directory=documents_root,
            index_directory=index_root,
            manifest=document_manifest,
        )
    )

    initial_components = (
        create_retrieval_components(
            config=config.retrieval,
            embedding_service=embedding_service,
            vector_store=vector_store,
        )
    )

    retrieval_components = (
        RetrievalComponentManager(
            initial_components
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
        retrieval_components=(
            retrieval_components
        ),
        prompt_builder=PromptBuilder(),
        answer_generator=answer_generator,
    )

    runtime = RAGRuntime(
        rag_service=rag_service,
        retrieval_components=(
            retrieval_components
        ),
        embedding_service=embedding_service,
        vector_store=vector_store,
        indexing_service=indexing_service,
        indexing_result=indexing_result,
        documents_directory=documents_root,
        index_directory=index_root,
        document_manifest=document_manifest,
        indexing_lock=threading.RLock(),
    )

    logger.info(
        "RAG runtime created | "
        "retriever=%s | reranker=%s | "
        "relevance_policy=%s | vectors=%d",
        config.retrieval.retriever,
        config.retrieval.reranker,
        config.retrieval.relevance_policy,
        vector_store.count,
    )

    return runtime


def reconfigure_retrieval(
    runtime: RAGRuntime,
    config: RetrievalConfig,
) -> RetrievalComponents:
    logger.info(
        "Reconfiguring retrieval | "
        "retriever=%s | reranker=%s | "
        "relevance_policy=%s",
        config.retriever,
        config.reranker,
        config.relevance_policy,
    )

    with runtime.indexing_lock:
        new_components = (
            create_retrieval_components(
                config=config,
                embedding_service=(
                    runtime.embedding_service
                ),
                vector_store=(
                    runtime.vector_store
                ),
            )
        )

        runtime.retrieval_components.replace(
            new_components
        )

    logger.info(
        "Retrieval switched | "
        "retriever=%s | reranker=%s | "
        "score_kind=%s",
        config.retriever,
        config.reranker,
        new_components.score_kind,
    )

    return new_components


def _create_embedding_service(
    config: Config,
) -> EmbeddingService:
    logger.info(
        "Creating embedding service | "
        "model=%s | device=%s | batch_size=%d",
        config.embedding.model,
        config.embedding.device,
        config.embedding.batch_size,
    )

    return LocalEmbeddingService(
        model_name=config.embedding.model,
        device=config.embedding.device,
        batch_size=config.embedding.batch_size,
    )


def _create_indexing_service(
    embedding_service: EmbeddingService,
) -> IndexingService:
    return IndexingService(
        loader=_create_document_loader(),
        chunker=_create_chunker(),
        embedding_service=embedding_service,
        vector_store_factory=FAISSVectorStore,
    )


def _create_document_loader() -> DocumentLoader:
    return MultiFormatDocumentLoader(
        strategies=[
            TextFileLoader(),
            PDFFileLoader(),
        ]
    )


def _create_chunker() -> Chunker:
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