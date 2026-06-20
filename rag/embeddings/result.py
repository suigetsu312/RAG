from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Float32Array = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    embedding: Float32Array
    latency_ms: float


@dataclass(frozen=True, slots=True)
class BatchEmbeddingResult:
    embeddings: Float32Array
    latency_ms: float