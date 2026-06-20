from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Self

import faiss
import numpy as np

from rag.document import Chunk, RetrievedChunk
from rag.vector_stores.base import Float32Array, VectorStore


class FAISSVectorStore(VectorStore):
    INDEX_FILE_NAME = "index.faiss"
    CHUNKS_FILE_NAME = "chunks.json"
    FORMAT_VERSION = 1

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than 0"
            )

        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[Chunk] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def count(self) -> int:
        return int(self._index.ntotal)

    def add_many(
        self,
        chunks: list[Chunk],
        embeddings: Float32Array,
    ) -> None:
        if not chunks:
            raise ValueError(
                "chunks must not be empty"
            )

        matrix = self._prepare_vectors(
            embeddings,
            name="embeddings",
        )

        if matrix.shape[0] != len(chunks):
            raise ValueError(
                "chunk count must match embedding count: "
                f"chunks={len(chunks)}, "
                f"embeddings={matrix.shape[0]}"
            )

        self._index.add(matrix)
        self._chunks.extend(chunks)

        if self.count != len(self._chunks):
            raise RuntimeError(
                "FAISS index and chunk metadata are inconsistent: "
                f"vectors={self.count}, "
                f"chunks={len(self._chunks)}"
            )

    def search(
        self,
        query_embedding: Float32Array,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        # 即使 index 為空，也先驗證 query 的格式。
        query_matrix = self._prepare_query(
            query_embedding
        )

        if self.count == 0:
            return []

        actual_k = min(top_k, self.count)

        scores, indices = self._index.search(
            query_matrix,
            actual_k,
        )

        results: list[RetrievedChunk] = []

        for score, index in zip(
            scores[0],
            indices[0],
            strict=True,
        ):
            if index < 0:
                continue

            results.append(
                RetrievedChunk(
                    chunk=self._chunks[int(index)],
                    score=float(score),
                )
            )

        return results

    def clear(self) -> None:
        self._index = faiss.IndexFlatIP(
            self._dimension
        )
        self._chunks.clear()

    def _prepare_query(
        self,
        query_embedding: Float32Array,
    ) -> Float32Array:
        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim != 1:
            raise ValueError(
                "query embedding must be a 1-D vector "
                "with shape (D,), "
                f"got shape={query.shape}"
            )

        return self._prepare_vectors(
            query.reshape(1, -1),
            name="query embedding",
        )

    def _prepare_vectors(
        self,
        vectors: Float32Array,
        *,
        name: str,
    ) -> Float32Array:
        matrix = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise ValueError(
                f"{name} must have shape (N, D), "
                f"got shape={matrix.shape}"
            )

        if matrix.shape[1] != self._dimension:
            raise ValueError(
                f"{name} dimension mismatch: "
                f"got={matrix.shape[1]}, "
                f"expected={self._dimension}"
            )

        # faiss.normalize_L2() 會原地修改矩陣，
        # 因此先建立獨立且連續的 float32 副本。
        matrix = np.ascontiguousarray(
            matrix.copy(),
            dtype=np.float32,
        )

        self._validate_nonzero_vectors(
            matrix,
            name=name,
        )

        faiss.normalize_L2(matrix)

        return matrix

    @staticmethod
    def _validate_nonzero_vectors(
        matrix: Float32Array,
        *,
        name: str,
    ) -> None:
        norms = np.linalg.norm(
            matrix,
            axis=1,
        )

        if np.any(norms == 0.0):
            raise ValueError(
                f"{name} must contain only non-zero vectors"
            )

    @staticmethod
    def _serialize_chunk(chunk: Chunk) -> dict[str, object]:
        return {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "text": chunk.text,
            "source": chunk.source,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
            "metadata": chunk.metadata,
        }

    @staticmethod
    def _deserialize_chunk(
        payload: dict[str, object],
    ) -> Chunk:
        metadata = payload.get("metadata", {})

        if not isinstance(metadata, dict):
            raise ValueError(
                "Chunk metadata must be a dictionary"
            )

        return Chunk(
            id=str(payload["id"]),
            document_id=str(payload["document_id"]),
            text=str(payload["text"]),
            source=str(payload["source"]),
            start_char=int(payload["start_char"]),
            end_char=int(payload["end_char"]),
            metadata=dict(metadata),
        )

    def save(
        self,
        directory: str | Path,
    ) -> None:
        target_directory = Path(directory)
        target_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = (
            target_directory / self.INDEX_FILE_NAME
        )
        chunks_path = (
            target_directory / self.CHUNKS_FILE_NAME
        )

        temporary_index_path = index_path.with_suffix(
            ".faiss.tmp"
        )
        temporary_chunks_path = chunks_path.with_suffix(
            ".json.tmp"
        )

        payload = {
            "version": self.FORMAT_VERSION,
            "dimension": self._dimension,
            "chunks": [
                self._serialize_chunk(chunk)
                for chunk in self._chunks
            ],
        }

        try:
            faiss.write_index(
                self._index,
                str(temporary_index_path),
            )

            temporary_chunks_path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            os.replace(
                temporary_index_path,
                index_path,
            )

            os.replace(
                temporary_chunks_path,
                chunks_path,
            )
        finally:
            temporary_index_path.unlink(
                missing_ok=True
            )
            temporary_chunks_path.unlink(
                missing_ok=True
            )

    @classmethod
    def load(
        cls,
        directory: str | Path,
    ) -> Self:
        source_directory = Path(directory)

        index_path = (
            source_directory / cls.INDEX_FILE_NAME
        )
        chunks_path = (
            source_directory / cls.CHUNKS_FILE_NAME
        )

        if not index_path.is_file():
            raise FileNotFoundError(
                f"FAISS index file does not exist: {index_path}"
            )

        if not chunks_path.is_file():
            raise FileNotFoundError(
                f"FAISS chunk metadata file does not exist: "
                f"{chunks_path}"
            )

        payload = json.loads(
            chunks_path.read_text(
                encoding="utf-8"
            )
        )

        version = payload.get("version")

        if version != cls.FORMAT_VERSION:
            raise ValueError(
                "Unsupported FAISS store format version: "
                f"got={version}, "
                f"expected={cls.FORMAT_VERSION}"
            )

        dimension = int(payload["dimension"])

        chunks = [
            cls._deserialize_chunk(item)
            for item in payload["chunks"]
        ]

        index = faiss.read_index(
            str(index_path)
        )

        if int(index.d) != dimension:
            raise ValueError(
                "FAISS index dimension does not match "
                "chunk metadata: "
                f"index={index.d}, "
                f"metadata={dimension}"
            )

        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(
                "FAISS index must use inner-product distance"
            )

        if int(index.ntotal) != len(chunks):
            raise ValueError(
                "FAISS index vector count does not match "
                "chunk metadata count: "
                f"vectors={index.ntotal}, "
                f"chunks={len(chunks)}"
            )

        store = cls(
            dimension=dimension
        )

        store._index = index
        store._chunks = chunks

        return store
