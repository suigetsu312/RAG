from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from rag.document import Chunk, RetrievedChunk


Float32Array = NDArray[np.float32]


class VectorStore(Protocol):
    @property
    def dimension(self) -> int:
        ...

    @property
    def count(self) -> int:
        ...

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        ...

    def add_many(
        self,
        chunks: list[Chunk],
        embeddings: Float32Array,
    ) -> None:
        ...

    def search(
        self,
        query_embedding: Float32Array,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        ...

    def clear(self) -> None:
        ...
