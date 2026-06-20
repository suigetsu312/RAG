from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

import rag.embeddings.local as local_module
from rag.embeddings import LocalEmbeddingService


class FakeSentenceTransformer:
    def __init__(
        self,
        model_name: str,
        *,
        device: str,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.calls: list[dict[str, Any]] = []

    def encode(
        self,
        sentences: str | list[str],
        *,
        batch_size: int | None = None,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        self.calls.append(
            {
                "sentences": sentences,
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )

        if isinstance(sentences, str):
            return np.array(
                [3.0, 4.0, 0.0],
                dtype=np.float64,
            )

        return np.array(
            [
                [1.0, 0.0, 0.0]
                for _ in sentences
            ],
            dtype=np.float64,
        )


@dataclass
class FakeModelFactory:
    created_models: list[FakeSentenceTransformer] = field(
        default_factory=list
    )

    def __call__(
        self,
        model_name: str,
        *,
        device: str,
    ) -> FakeSentenceTransformer:
        model = FakeSentenceTransformer(
            model_name,
            device=device,
        )

        self.created_models.append(model)
        return model


@pytest.fixture
def model_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> FakeModelFactory:
    factory = FakeModelFactory()

    monkeypatch.setattr(
        local_module,
        "SentenceTransformer",
        factory,
    )

    return factory


def test_initializes_sentence_transformer_with_configuration(
    model_factory: FakeModelFactory,
) -> None:
    service = LocalEmbeddingService(
        model_name="test-model",
        device="cpu",
        batch_size=8,
    )

    assert len(model_factory.created_models) == 1

    model = model_factory.created_models[0]

    assert model.model_name == "test-model"
    assert model.device == "cpu"

    assert service.model_name == "test-model"
    assert service.device == "cpu"
    assert service.batch_size == 8


@pytest.mark.parametrize(
    "batch_size",
    [
        0,
        -1,
        -10,
    ],
)
def test_rejects_invalid_batch_size(
    model_factory: FakeModelFactory,
    batch_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="batch_size must be greater than 0",
    ):
        LocalEmbeddingService(
            model_name="test-model",
            device="cpu",
            batch_size=batch_size,
        )

    assert model_factory.created_models == []


def test_embed_returns_float32_contiguous_vector(
    model_factory: FakeModelFactory,
) -> None:
    service = LocalEmbeddingService(
        model_name="test-model",
        device="cpu",
        batch_size=8,
    )

    result = service.embed("hello")

    assert result.embedding.shape == (3,)
    assert result.embedding.dtype == np.float32
    assert result.embedding.flags.c_contiguous
    assert result.latency_ms >= 0.0

    np.testing.assert_allclose(
        result.embedding,
        np.array(
            [3.0, 4.0, 0.0],
            dtype=np.float32,
        ),
    )


def test_embed_uses_normalized_numpy_output(
    model_factory: FakeModelFactory,
) -> None:
    service = LocalEmbeddingService(
        model_name="test-model",
        device="cpu",
        batch_size=8,
    )

    service.embed("hello")

    model = model_factory.created_models[0]
    call = model.calls[0]

    assert call["sentences"] == "hello"
    assert call["batch_size"] is None
    assert call["normalize_embeddings"] is True
    assert call["convert_to_numpy"] is True
    assert call["show_progress_bar"] is False


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\n",
        "\t",
    ],
)
def test_embed_rejects_empty_text(
    model_factory: FakeModelFactory,
    text: str,
) -> None:
    service = LocalEmbeddingService(
        model_name="test-model",
        device="cpu",
        batch_size=8,
    )

    with pytest.raises(
        ValueError,
        match="text must not be empty",
    ):
        service.embed(text)

    model = model_factory.created_models[0]
    assert model.calls == []


def test_embed_batch_returns_float32_contiguous_matrix(
    model_factory: FakeModelFactory,
) -> None:
    service = LocalEmbeddingService(
        model_name="test-model",
        device="cpu",
        batch_size=2,
    )

    result = service.embed_batch(
        [
            "first",
            "second",
            "third",
        ]
    )

    assert result.embeddings.shape == (3, 3)
    assert result.embeddings.dtype == np.float32
    assert result.embeddings.flags.c_contiguous
    assert result.latency_ms >= 0.0

    expected = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        result.embeddings,
        expected,
    )


def test_embed_batch_uses_configured_batch_size(
    model_factory: FakeModelFactory,
) -> None:
    service = LocalEmbeddingService(
        model_name="test-model",
        device="cpu",
        batch_size=16,
    )

    texts = [
        "first",
        "second",
    ]

    service.embed_batch(texts)

    model = model_factory.created_models[0]
    call = model.calls[0]

    assert call["sentences"] == texts
    assert call["batch_size"] == 16
    assert call["normalize_embeddings"] is True
    assert call["convert_to_numpy"] is True
    assert call["show_progress_bar"] is False


def test_embed_batch_rejects_empty_list(
    model_factory: FakeModelFactory,
) -> None:
    service = LocalEmbeddingService(
        model_name="test-model",
        device="cpu",
        batch_size=8,
    )

    with pytest.raises(
        ValueError,
        match="texts must not be empty",
    ):
        service.embed_batch([])

    model = model_factory.created_models[0]
    assert model.calls == []


@pytest.mark.parametrize(
    "texts",
    [
        ["valid", ""],
        ["valid", "   "],
        ["\n"],
    ],
)
def test_embed_batch_rejects_empty_items(
    model_factory: FakeModelFactory,
    texts: list[str],
) -> None:
    service = LocalEmbeddingService(
        model_name="test-model",
        device="cpu",
        batch_size=8,
    )

    with pytest.raises(
        ValueError,
        match="texts must not contain empty strings",
    ):
        service.embed_batch(texts)

    model = model_factory.created_models[0]
    assert model.calls == []


def test_embedding_dimension_is_read_from_single_result_shape(
    model_factory: FakeModelFactory,
) -> None:
    service = LocalEmbeddingService(
        model_name="test-model",
        device="cpu",
        batch_size=8,
    )

    result = service.embed("dimension probe")

    dimension = result.embedding.shape[0]

    assert dimension == 3


def test_embedding_dimension_is_read_from_batch_result_shape(
    model_factory: FakeModelFactory,
) -> None:
    service = LocalEmbeddingService(
        model_name="test-model",
        device="cpu",
        batch_size=8,
    )

    result = service.embed_batch(
        [
            "first",
            "second",
        ]
    )

    dimension = result.embeddings.shape[1]

    assert dimension == 3