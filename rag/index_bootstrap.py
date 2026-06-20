from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from rag.document_manifest import (
    DocumentManifest,
    DocumentRecord,
    calculate_sha256,
)
from rag.embeddings import EmbeddingService
from rag.indexing_service import (
    IndexingResult,
    IndexingService,
)
from rag.vector_stores import FAISSVectorStore


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
                document_count=indexed_source.document_count,
                chunk_count=indexed_source.chunk_count,
                created_at=created_at,
            )
        )

    manifest.replace_all(records)


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
    documents_root = Path(documents_directory)
    index_root = Path(index_directory)

    index_path = (
        index_root
        / FAISSVectorStore.INDEX_FILE_NAME
    )
    chunks_path = (
        index_root
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
        documents_root
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
        vector_store = FAISSVectorStore.load(
            index_root
        )

        logger.info(
            "FAISS index loaded | "
            "documents=%d | vectors=%d | dimension=%d",
            len(current_source_hashes),
            vector_store.count,
            vector_store.dimension,
        )

        return vector_store, None

    document_files = find_document_files(
        documents_root
    )

    if not document_files:
        logger.info(
            "No source documents found; "
            "creating empty index"
        )

        vector_store = FAISSVectorStore(
            embedding_service.dimension
        )

        vector_store.save(index_root)
        manifest.replace_all([])

        return vector_store, None

    rebuild_reason = (
        "source files changed"
        if index_exists
        else "index does not exist"
    )

    logger.info(
        "Building FAISS index | files=%d | reason=%s",
        len(document_files),
        rebuild_reason,
    )

    indexing_result = (
        indexing_service.build_directory(
            documents_root
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

    vector_store.save(index_root)

    update_manifest_from_indexing_result(
        manifest=manifest,
        indexing_result=indexing_result,
        documents_directory=documents_root,
    )

    logger.info(
        "FAISS index built | "
        "files=%d | documents=%d | "
        "chunks=%d | dimension=%d",
        len(document_files),
        indexing_result.document_count,
        indexing_result.chunk_count,
        indexing_result.embedding_dimension,
    )

    return vector_store, indexing_result