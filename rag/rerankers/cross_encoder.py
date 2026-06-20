from __future__ import annotations

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from rag.document import RetrievedChunk


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than 0"
            )

        if max_length <= 0:
            raise ValueError(
                "max_length must be greater than 0"
            )

        self._device = torch.device(device)
        self._batch_size = batch_size
        self._max_length = max_length

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self._model = (
            AutoModelForSequenceClassification
            .from_pretrained(model_name)
        )

        self._model.to(self._device)
        self._model.eval()

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must not be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not candidates:
            return []

        scores: list[float] = []

        for start in range(
            0,
            len(candidates),
            self._batch_size,
        ):
            batch = candidates[
                start:start + self._batch_size
            ]

            queries = [
                normalized_query
                for _ in batch
            ]

            passages = [
                item.chunk.text
                for item in batch
            ]

            inputs = self._tokenizer(
                queries,
                passages,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )

            inputs = {
                name: tensor.to(self._device)
                for name, tensor in inputs.items()
            }

            with torch.inference_mode():
                logits = self._model(
                    **inputs,
                    return_dict=True,
                ).logits.view(-1).float()

                normalized_scores = torch.sigmoid(
                    logits
                )

            scores.extend(
                normalized_scores.cpu().tolist()
            )

        reranked = [
            RetrievedChunk(
                chunk=item.chunk,
                score=float(score),
            )
            for item, score in zip(
                candidates,
                scores,
                strict=True,
            )
        ]

        reranked.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return reranked[:top_k]