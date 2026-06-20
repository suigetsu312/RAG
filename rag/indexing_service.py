from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from rag.chunkers import Chunker
from rag.document import Chunk, Document
from rag.embeddings import EmbeddingService
from rag.loaders import DocumentLoader
from rag.metrics import IndexingTimings
from rag.vector_stores import VectorStore

VectorStoreFactory = Callable[[int], VectorStore]


@dataclass(frozen=True, slots=True)
class IndexedSource:
    source: str
    document_count: int
    chunk_count: int


@dataclass(frozen=True, slots=True)
class IndexingResult:
    vector_store: VectorStore
    document_count: int
    chunk_count: int
    embedding_dimension: int
    indexed_sources: list[IndexedSource]
    timings: IndexingTimings


@dataclass(frozen=True, slots=True)
class IndexingUpdateResult:
    document_count: int
    chunk_count: int
    timings: IndexingTimings


class IndexingService:
    def __init__(
        self,
        loader: DocumentLoader,
        chunker: Chunker,
        embedding_service: EmbeddingService,
        vector_store_factory: VectorStoreFactory,
    ) -> None:
        self._loader = loader
        self._chunker = chunker
        self._embedding_service = embedding_service
        self._vector_store_factory = vector_store_factory

    def build(self, directory: str | Path) -> IndexingResult:
        return self.build_directory(directory)

    def build_directory(
        self,
        directory: str | Path,
    ) -> IndexingResult:
        return self._build(
            lambda: self._loader.load_directory(directory)
        )

    def build_file(
        self,
        path: str | Path,
    ) -> IndexingResult:
        return self._build(
            lambda: self._loader.load_file(path)
        )

    def _build(
        self,
        load_documents: Callable[[], list[Document]],
    ) -> IndexingResult:
        total_started_at = perf_counter()

        load_started_at = perf_counter()
        documents = load_documents()
        document_load_ms = self._elapsed_ms(load_started_at)

        chunking_started_at = perf_counter()
        chunks = self._create_chunks(documents)
        chunking_ms = self._elapsed_ms(chunking_started_at)

        if not chunks:
            raise ValueError(
                "No chunks were generated from the input documents"
            )

        document_counts = Counter(
            document.source
            for document in documents
        )
        chunk_counts = Counter(
            chunk.source
            for chunk in chunks
        )
        indexed_sources = [
            IndexedSource(
                source=source,
                document_count=document_count,
                chunk_count=chunk_counts.get(source, 0),
            )
            for source, document_count
            in sorted(document_counts.items())
        ]

        embedding_started_at = perf_counter()
        embedding_result = self._embedding_service.embed_batch(
            [chunk.text for chunk in chunks]
        )
        embedding_ms = self._elapsed_ms(embedding_started_at)

        embeddings = np.asarray(
            embedding_result.embeddings,
            dtype=np.float32,
        )

        self._validate_embeddings(
            embeddings=embeddings,
            expected_count=len(chunks),
        )

        embedding_dimension = int(
            embeddings.shape[1]
        )

        vector_store = self._vector_store_factory(
            embedding_dimension
        )

        if vector_store.dimension != embedding_dimension:
            raise RuntimeError(
                "Vector store dimension does not match "
                "embedding dimension: "
                f"store={vector_store.dimension}, "
                f"embedding={embedding_dimension}"
            )

        vector_store_started_at = perf_counter()
        vector_store.add_many(
            chunks=chunks,
            embeddings=embeddings,
        )
        vector_store_add_ms = self._elapsed_ms(
            vector_store_started_at
        )

        return IndexingResult(
            vector_store=vector_store,
            document_count=len(documents),
            chunk_count=len(chunks),
            embedding_dimension=embedding_dimension,
            indexed_sources=indexed_sources,
            timings=IndexingTimings(
                document_load_ms=document_load_ms,
                chunking_ms=chunking_ms,
                embedding_ms=embedding_ms,
                vector_store_add_ms=vector_store_add_ms,
                total_ms=self._elapsed_ms(total_started_at),
            ),
        )

    def _create_chunks(
        self,
        documents: list[Document],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        for document in documents:
            chunks.extend(
                self._chunker.split(document)
            )

        return chunks

    @staticmethod
    def _validate_embeddings(
        embeddings: np.ndarray,
        expected_count: int,
    ) -> None:
        if embeddings.ndim != 2:
            raise RuntimeError(
                "Embedding service must return a matrix "
                "with shape (N, D), "
                f"got shape={embeddings.shape}"
            )

        if embeddings.shape[0] != expected_count:
            raise RuntimeError(
                "Embedding count does not match chunk count: "
                f"embeddings={embeddings.shape[0]}, "
                f"chunks={expected_count}"
            )

        if embeddings.shape[1] <= 0:
            raise RuntimeError(
                "Embedding dimension must be greater than 0"
            )

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (
            perf_counter() - started_at
        ) * 1000.0

    def add_file(
        self,
        path: str | Path,
        vector_store: VectorStore,
    ) -> IndexingUpdateResult:
        total_started_at = perf_counter()

        load_started_at = perf_counter()
        documents = self._loader.load_file(path)
        document_load_ms = self._elapsed_ms(load_started_at)

        chunking_started_at = perf_counter()
        chunks = self._create_chunks(documents)
        chunking_ms = self._elapsed_ms(chunking_started_at)

        if not chunks:
            raise ValueError(
                "No chunks were generated from the input file"
            )

        embedding_started_at = perf_counter()
        embedding_result = self._embedding_service.embed_batch(
            [chunk.text for chunk in chunks]
        )
        embedding_ms = self._elapsed_ms(embedding_started_at)

        embeddings = np.asarray(
            embedding_result.embeddings,
            dtype=np.float32,
        )

        self._validate_embeddings(
            embeddings=embeddings,
            expected_count=len(chunks),
        )

        embedding_dimension = int(embeddings.shape[1])

        if embedding_dimension != vector_store.dimension:
            raise RuntimeError(
                "Embedding dimension does not match vector store: "
                f"embedding={embedding_dimension}, "
                f"store={vector_store.dimension}"
            )

        add_started_at = perf_counter()

        vector_store.add_many(
            chunks=chunks,
            embeddings=embeddings,
        )

        vector_store_add_ms = self._elapsed_ms(
            add_started_at
        )

        return IndexingUpdateResult(
            document_count=len(documents),
            chunk_count=len(chunks),
            timings=IndexingTimings(
                document_load_ms=document_load_ms,
                chunking_ms=chunking_ms,
                embedding_ms=embedding_ms,
                vector_store_add_ms=vector_store_add_ms,
                total_ms=self._elapsed_ms(total_started_at),
            ),
        )
