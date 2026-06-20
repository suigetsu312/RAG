from __future__ import annotations

import numpy as np
import pytest

from rag.document import Chunk
from rag.vector_stores import FAISSVectorStore


def make_chunk(
    chunk_id: str,
    text: str,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="doc-1",
        text=text,
        source="memory",
        start_char=0,
        end_char=len(text),
    )


def test_store_starts_empty() -> None:
    store = FAISSVectorStore(dimension=3)

    assert store.dimension == 3
    assert store.count == 0


def test_add_many_adds_chunks_and_vectors() -> None:
    store = FAISSVectorStore(dimension=3)

    chunks = [
        make_chunk(
            "chunk-0",
            "transformer",
        ),
        make_chunk(
            "chunk-1",
            "convolution",
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store.add_many(
        chunks=chunks,
        embeddings=embeddings,
    )

    assert store.count == 2


def test_search_returns_most_similar_chunk() -> None:
    store = FAISSVectorStore(dimension=3)

    chunks = [
        make_chunk(
            "transformer",
            "self-attention",
        ),
        make_chunk(
            "cnn",
            "convolution",
        ),
        make_chunk(
            "faiss",
            "vector search",
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store.add_many(
        chunks=chunks,
        embeddings=embeddings,
    )

    query = np.array(
        [0.9, 0.1, 0.0],
        dtype=np.float32,
    )

    results = store.search(
        query_embedding=query,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.id == "transformer"
    assert results[1].chunk.id == "cnn"
    assert results[0].score > results[1].score


def test_search_uses_cosine_similarity() -> None:
    store = FAISSVectorStore(dimension=2)

    chunks = [
        make_chunk(
            "same-direction",
            "same",
        ),
        make_chunk(
            "different-direction",
            "different",
        ),
    ]

    embeddings = np.array(
        [
            [10.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    store.add_many(
        chunks=chunks,
        embeddings=embeddings,
    )

    query = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    results = store.search(
        query_embedding=query,
        top_k=2,
    )

    assert results[0].chunk.id == "same-direction"

    assert results[0].score == pytest.approx(
        1.0,
        abs=1e-6,
    )

    assert results[1].score == pytest.approx(
        1.0 / np.sqrt(2.0),
        abs=1e-6,
    )


def test_search_limits_top_k_to_store_count() -> None:
    store = FAISSVectorStore(dimension=2)

    chunks = [
        make_chunk(
            "chunk-0",
            "first",
        ),
        make_chunk(
            "chunk-1",
            "second",
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    store.add_many(
        chunks=chunks,
        embeddings=embeddings,
    )

    results = store.search(
        query_embedding=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        top_k=10,
    )

    assert len(results) == 2


def test_empty_store_returns_empty_results() -> None:
    store = FAISSVectorStore(dimension=3)

    results = store.search(
        query_embedding=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        top_k=5,
    )

    assert results == []


def test_add_many_rejects_empty_chunks() -> None:
    store = FAISSVectorStore(dimension=3)

    with pytest.raises(
        ValueError,
        match="chunks must not be empty",
    ):
        store.add_many(
            chunks=[],
            embeddings=np.empty(
                (0, 3),
                dtype=np.float32,
            ),
        )


def test_add_many_rejects_row_count_mismatch() -> None:
    store = FAISSVectorStore(dimension=3)

    chunks = [
        make_chunk(
            "chunk-0",
            "first",
        ),
        make_chunk(
            "chunk-1",
            "second",
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="chunk count must match embedding count",
    ):
        store.add_many(
            chunks=chunks,
            embeddings=embeddings,
        )

    assert store.count == 0


def test_add_many_rejects_dimension_mismatch() -> None:
    store = FAISSVectorStore(dimension=3)

    chunks = [
        make_chunk(
            "chunk-0",
            "first",
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="embeddings dimension mismatch",
    ):
        store.add_many(
            chunks=chunks,
            embeddings=embeddings,
        )

    assert store.count == 0


def test_search_rejects_dimension_mismatch() -> None:
    store = FAISSVectorStore(dimension=3)

    with pytest.raises(
        ValueError,
        match="query embedding dimension mismatch",
    ):
        store.search(
            query_embedding=np.array(
                [1.0, 0.0],
                dtype=np.float32,
            ),
            top_k=1,
        )


def test_search_rejects_matrix_query() -> None:
    store = FAISSVectorStore(dimension=3)

    with pytest.raises(
        ValueError,
        match="query embedding must be a 1-D vector",
    ):
        store.search(
            query_embedding=np.array(
                [[1.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            top_k=1,
        )


def test_add_many_rejects_zero_norm_vector() -> None:
    store = FAISSVectorStore(dimension=3)

    chunks = [
        make_chunk(
            "chunk-0",
            "zero",
        ),
    ]

    embeddings = np.zeros(
        (1, 3),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match=(
            "embeddings must contain only non-zero vectors"
        ),
    ):
        store.add_many(
            chunks=chunks,
            embeddings=embeddings,
        )


def test_search_rejects_zero_norm_query() -> None:
    store = FAISSVectorStore(dimension=3)

    with pytest.raises(
        ValueError,
        match=(
            "query embedding must contain only non-zero vectors"
        ),
    ):
        store.search(
            query_embedding=np.zeros(
                3,
                dtype=np.float32,
            ),
            top_k=1,
        )


def test_clear_removes_all_vectors() -> None:
    store = FAISSVectorStore(dimension=3)

    chunks = [
        make_chunk(
            "chunk-0",
            "first",
        ),
        make_chunk(
            "chunk-1",
            "second",
        ),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store.add_many(
        chunks=chunks,
        embeddings=embeddings,
    )

    assert store.count == 2

    store.clear()

    assert store.count == 0

    results = store.search(
        query_embedding=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
        top_k=5,
    )

    assert results == []


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_search_rejects_invalid_top_k(
    top_k: int,
) -> None:
    store = FAISSVectorStore(dimension=3)

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        store.search(
            query_embedding=np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "dimension",
    [
        0,
        -1,
    ],
)
def test_rejects_invalid_dimension(
    dimension: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="dimension must be greater than 0",
    ):
        FAISSVectorStore(
            dimension=dimension
        )