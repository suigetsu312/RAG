from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from rag.embeddings.base import EmbeddingService
from rag.embeddings.result import (
    BatchEmbeddingResult,
    EmbeddingResult,
)

class LocalEmbeddingService(EmbeddingService):
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self._model = SentenceTransformer(
            model_name,
            device=device,
        )

    def embed(self, text: str) -> EmbeddingResult:
        if not text.strip():
            raise ValueError("text must not be empty")

        start_time = time.perf_counter()

        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embedding = np.ascontiguousarray(
            embedding,
            dtype=np.float32,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        return EmbeddingResult(
            embedding=embedding,
            latency_ms=latency_ms,
        )

    def embed_batch(
        self,
        texts: list[str],
    ) -> BatchEmbeddingResult:
        if not texts:
            raise ValueError("texts must not be empty")

        if any(not text.strip() for text in texts):
            raise ValueError(
                "texts must not contain empty strings"
            )

        start_time = time.perf_counter()

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        return BatchEmbeddingResult(
            embeddings=embeddings,
            latency_ms=latency_ms,
        )