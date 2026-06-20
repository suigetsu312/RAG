from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from rag.chunkers.base import Chunker
from rag.embeddings.base import EmbeddingService
from rag.loaders.base import DocumentLoader
from rag.metrics import IndexingTimings
from rag.vector_stores.base import VectorStore

VectorStoreFactory = Callable[[int], VectorStore]


@dataclass(frozen=True, slots=True)
class IndexingResult:
    vector_store: VectorStore
    document_count: int
    chunk_count: int
    embedding_dimension: int
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

    def build(
        self,
        directory: str | Path,
    ) -> IndexingResult:
        total_start = perf_counter()

        load_start = perf_counter()
        documents = self._loader.load_directory(directory)
        document_load_ms = (perf_counter() - load_start) * 1000.0

        chunk_start = perf_counter()
        chunks = [
            chunk
            for document in documents
            for chunk in self._chunker.split(document)
        ]

        if not chunks:
            raise ValueError(
                "No chunks were generated from the document directory"
            )
        chunking_ms = (perf_counter() - chunk_start) * 1000.0

        embedding_start = perf_counter()
        result = self._embedding_service.embed_batch(
            [chunk.text for chunk in chunks]
        )

        embedding_ms = (perf_counter() - embedding_start) * 1000.0

        embeddings = np.asarray(
            result.embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise RuntimeError(
                "Embedding service must return shape (N, D)"
            )

        if embeddings.shape[0] != len(chunks):
            raise RuntimeError(
                "Embedding count does not match chunk count"
            )

        store_start = perf_counter()

        dimension = int(embeddings.shape[1])

        vector_store = self._vector_store_factory(
            dimension
        )

        vector_store.add_many(
            chunks,
            embeddings,
        )

        vector_store_add_ms = (perf_counter() - store_start) * 1000.0

        total_ms = (perf_counter() - total_start) * 1000.0

        return IndexingResult(
            vector_store=vector_store,
            document_count=len(documents),
            chunk_count=len(chunks),
            embedding_dimension=dimension,
            timings=IndexingTimings(
                document_load_ms=document_load_ms,
                chunking_ms=chunking_ms,
                embedding_ms=embedding_ms,
                vector_store_add_ms=vector_store_add_ms,
                total_ms=total_ms,
            ),
        )
