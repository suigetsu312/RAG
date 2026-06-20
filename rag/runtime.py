from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import Config, GenerationOptions
from rag.chunkers import (
    Chunker,
    FixedSizeChunker,
    RoutingChunker,
)
from rag.document_manifest import (
    DocumentManifest,
    DocumentRecord,
    calculate_sha256,
)
from rag.embeddings import (
    EmbeddingService,
    LocalEmbeddingService,
)
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

SUPPORTED_DOCUMENT_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
}


def find_document_files(
    directory: str | Path,
) -> list[Path]:
    root = Path(directory)

    if not root.exists():
        return []

    return sorted(
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_DOCUMENT_SUFFIXES
        )
    )


def calculate_source_hashes(
    directory: str | Path,
) -> dict[str, str]:
    root = Path(directory)

    return {
        path.relative_to(root).as_posix(): calculate_sha256(path)
        for path in find_document_files(root)
    }


@dataclass(frozen=True, slots=True)
class RAGRuntime:
    rag_service: RAGService
    vector_store: FAISSVectorStore
    indexing_service: IndexingService
    indexing_result: IndexingResult | None
    documents_directory: Path
    index_directory: Path
    document_manifest: DocumentManifest
    indexing_lock: threading.RLock


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


def bootstrap_vector_store(
    *,
    indexing_service: IndexingService,
    embedding_service: EmbeddingService,
    documents_directory: str | Path,
    index_directory: str | Path,
    manifest: DocumentManifest,
) -> tuple[
    FAISSVectorStore,
    IndexingResult | None,
]:
    documents_directory = Path(documents_directory)
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
        logger.warning(
            "Incomplete FAISS persistence detected; "
            "rebuilding index"
        )
        index_exists = False
        chunks_exist = False

    current_source_hashes = calculate_source_hashes(
        documents_directory
    )
    manifest_source_hashes = manifest.source_hashes()

    sources_match = (
        current_source_hashes
        == manifest_source_hashes
    )

    if (
        index_exists
        and chunks_exist
        and sources_match
    ):
        logger.info(
            "Loading existing FAISS index | documents=%d",
            len(current_source_hashes),
        )

        return (
            FAISSVectorStore.load(index_directory),
            None,
        )

    document_files = find_document_files(
        documents_directory
    )

    if not document_files:
        logger.info(
            "No source documents found; creating empty index"
        )

        vector_store = FAISSVectorStore(
            embedding_service.dimension
        )

        vector_store.save(index_directory)
        manifest.replace_all([])

        return vector_store, None

    logger.info(
        "Building FAISS index from source documents | "
        "files=%d | reason=%s",
        len(document_files),
        (
            "source files changed"
            if index_exists
            else "index does not exist"
        ),
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
            "IndexingService returned an unsupported "
            "vector store type"
        )

    vector_store.save(index_directory)

    update_manifest_from_indexing_result(
        manifest=manifest,
        indexing_result=indexing_result,
        documents_directory=documents_directory,
    )

    logger.info(
        "FAISS index built | files=%d | "
        "documents=%d | chunks=%d",
        len(document_files),
        indexing_result.document_count,
        indexing_result.chunk_count,
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

    documents_directory = Path(documents_directory)
    index_directory = Path(index_directory)

    documents_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    index_directory.mkdir(
        parents=True,
        exist_ok=True,
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

    document_manifest = DocumentManifest.load(
        index_directory / "documents.json"
    )

    vector_store, indexing_result = (
        bootstrap_vector_store(
            indexing_service=indexing_service,
            embedding_service=embedding_service,
            documents_directory=documents_directory,
            index_directory=index_directory,
            manifest=document_manifest,
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
        document_manifest=document_manifest,
        indexing_lock=threading.RLock(),
    )


def update_manifest_from_indexing_result(
    *,
    manifest: DocumentManifest,
    indexing_result: IndexingResult,
    documents_directory: str | Path,
) -> None:
    root = Path(documents_directory)
    created_at = datetime.now(timezone.utc).isoformat()

    records: list[DocumentRecord] = []

    for indexed_source in indexing_result.indexed_sources:
        path = root / indexed_source.source

        if not path.is_file():
            raise RuntimeError(
                "Indexed source file does not exist: "
                f"{indexed_source.source}"
            )

        sha256 = calculate_sha256(path)

        records.append(
            DocumentRecord(
                id=sha256,
                file_name=path.name,
                source=indexed_source.source,
                sha256=sha256,
                size_bytes=path.stat().st_size,
                document_count=(
                    indexed_source.document_count
                ),
                chunk_count=indexed_source.chunk_count,
                created_at=created_at,
            )
        )

    manifest.replace_all(records)
