from pathlib import Path

import numpy as np
import pytest

from rag.document import Chunk, Document
from rag.embeddings import BatchEmbeddingResult
from rag.indexing_service import IndexingService


class FakeLoader:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.received_directory: str | Path | None = None

    def load_directory(
        self,
        directory: str | Path,
    ) -> list[Document]:
        self.received_directory = directory
        return self.documents


class FakeChunker:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def split(self, document: Document) -> list[Chunk]:
        return [
            chunk
            for chunk in self.chunks
            if chunk.document_id == document.id
        ]


class FakeEmbeddingService:
    def __init__(self, embeddings: np.ndarray) -> None:
        self.embeddings = embeddings
        self.received_texts: list[str] | None = None

    def embed(self, text: str):
        raise NotImplementedError

    def embed_batch(
        self,
        texts: list[str],
    ) -> BatchEmbeddingResult:
        self.received_texts = texts
        return BatchEmbeddingResult(
            embeddings=self.embeddings,
            latency_ms=1.0,
        )


class FakeVectorStore:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.added_chunks: list[Chunk] = []
        self.added_embeddings: np.ndarray | None = None

    @property
    def count(self) -> int:
        return len(self.added_chunks)

    def add_many(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ) -> None:
        self.added_chunks = chunks
        self.added_embeddings = embeddings

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        raise NotImplementedError

    def clear(self) -> None:
        self.added_chunks.clear()


def test_build_returns_vector_store_and_timings() -> None:
    document = Document(
        id="doc-1",
        text="first second",
        source="doc.txt",
    )
    chunks = [
        Chunk(
            id="doc-1:chunk:0",
            document_id=document.id,
            text="first",
            source=document.source,
            start_char=0,
            end_char=5,
        ),
        Chunk(
            id="doc-1:chunk:1",
            document_id=document.id,
            text="second",
            source=document.source,
            start_char=6,
            end_char=12,
        ),
    ]
    loader = FakeLoader([document])
    embedding_service = FakeEmbeddingService(
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
    )

    service = IndexingService(
        loader=loader,
        chunker=FakeChunker(chunks),
        embedding_service=embedding_service,
        vector_store_factory=FakeVectorStore,
    )

    result = service.build("documents")

    assert loader.received_directory == "documents"
    assert embedding_service.received_texts == ["first", "second"]
    assert result.document_count == 1
    assert result.chunk_count == 2
    assert result.embedding_dimension == 3
    assert result.vector_store.count == 2
    assert result.timings.total_ms >= 0.0
    assert result.timings.embedding_ms >= 0.0


def test_build_rejects_documents_without_chunks() -> None:
    service = IndexingService(
        loader=FakeLoader(
            [Document(id="doc-1", text="", source="empty.txt")]
        ),
        chunker=FakeChunker([]),
        embedding_service=FakeEmbeddingService(
            np.empty((0, 3), dtype=np.float32)
        ),
        vector_store_factory=FakeVectorStore,
    )

    with pytest.raises(
        ValueError,
        match="No chunks were generated",
    ):
        service.build("documents")
