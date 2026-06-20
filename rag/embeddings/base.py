from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


Float32Array = NDArray[np.float32]


class EmbeddingResultProtocol(Protocol):
    embedding: Float32Array
    latency_ms: float


class BatchEmbeddingResultProtocol(Protocol):
    embeddings: Float32Array
    latency_ms: float


class EmbeddingService(Protocol):
    def embed(
        self,
        text: str,
    ) -> EmbeddingResultProtocol:
        ...

    def embed_batch(
        self,
        texts: list[str],
    ) -> BatchEmbeddingResultProtocol:
        ...