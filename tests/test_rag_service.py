from __future__ import annotations

import numpy as np
import pytest

from rag.document import (
    Chunk,
    RAGResult,
    RetrievedChunk,
)
from rag.embeddings import EmbeddingResult
from rag.generators.result import GenerationResult
from rag.prompts import PromptBuilder
from rag.rag_service import RAGService


class FakeEmbeddingService:
    def __init__(
        self,
        embedding: np.ndarray,
    ) -> None:
        self._embedding = embedding
        self.received_text: str | None = None

    def embed(
        self,
        text: str,
    ) -> EmbeddingResult:
        self.received_text = text

        return EmbeddingResult(
            embedding=self._embedding,
            latency_ms=1.5,
        )

    def embed_batch(
        self,
        texts: list[str],
    ):
        raise NotImplementedError


class FakeVectorStore:
    def __init__(
        self,
        results: list[RetrievedChunk],
        dimension: int = 3,
    ) -> None:
        self._results = results
        self._dimension = dimension
        self.received_query: np.ndarray | None = None
        self.received_top_k: int | None = None

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def count(self) -> int:
        return len(self._results)

    def add_many(
        self,
        chunks,
        embeddings,
    ) -> None:
        raise NotImplementedError

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        self.received_query = query_embedding
        self.received_top_k = top_k

        return self._results[:top_k]

    def clear(self) -> None:
        self._results.clear()


class FakeAnswerGenerator:
    def __init__(
        self,
        answer: str,
    ) -> None:
        self._answer = answer
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None
        self.call_count = 0

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> GenerationResult:
        self.call_count += 1
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        return GenerationResult(
            content=self._answer,
            metadata={"latency_ms": 12.5},
        )


def make_retrieved_chunk() -> RetrievedChunk:
    text = "Transformer 使用 self-attention 處理序列。"

    chunk = Chunk(
        id="transformer:chunk:0",
        document_id="transformer",
        text=text,
        source="ai/transformer.txt",
        start_char=0,
        end_char=len(text),
        metadata={
            "file_type": "text",
        },
    )

    return RetrievedChunk(
        chunk=chunk,
        score=0.95,
    )


def test_ask_runs_complete_rag_pipeline() -> None:
    query_embedding = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )

    retrieved = make_retrieved_chunk()

    embedding_service = FakeEmbeddingService(
        embedding=query_embedding,
    )

    vector_store = FakeVectorStore(
        results=[retrieved],
    )

    answer_generator = FakeAnswerGenerator(
        answer="Transformer 使用 self-attention。[Source 1]"
    )

    service = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        prompt_builder=PromptBuilder(),
        answer_generator=answer_generator,
    )

    result = service.ask(
        question="  Transformer 如何處理序列？  ",
        top_k=3,
    )

    assert isinstance(result, RAGResult)

    assert result.answer == (
        "Transformer 使用 self-attention。[Source 1]"
    )

    assert result.retrieved_chunks == [retrieved]
    assert result.generation_metadata == {
        "latency_ms": 12.5,
    }

    assert embedding_service.received_text == (
        "Transformer 如何處理序列？"
    )

    assert vector_store.received_top_k == 3

    np.testing.assert_array_equal(
        vector_store.received_query,
        query_embedding,
    )

    assert answer_generator.call_count == 1

    assert answer_generator.system_prompt is not None
    assert "只能根據" in (
        answer_generator.system_prompt
    )

    assert answer_generator.user_prompt is not None
    assert "Transformer 如何處理序列？" in (
        answer_generator.user_prompt
    )
    assert "self-attention" in (
        answer_generator.user_prompt
    )


def test_retrieve_returns_chunks_without_generating_answer() -> None:
    query_embedding = np.array(
        [1.0, 0.0, 0.0],
        dtype=np.float32,
    )
    retrieved = make_retrieved_chunk()
    vector_store = FakeVectorStore(results=[retrieved])
    answer_generator = FakeAnswerGenerator(
        answer="should not be used",
    )
    service = RAGService(
        embedding_service=FakeEmbeddingService(
            embedding=query_embedding,
        ),
        vector_store=vector_store,
        prompt_builder=PromptBuilder(),
        answer_generator=answer_generator,
    )

    result = service.retrieve(
        query="  Transformer  ",
        top_k=2,
    )

    assert result.retrieved_chunks == [retrieved]
    assert result.timings.query_embedding_ms >= 0.0
    assert result.timings.retrieval_ms >= 0.0
    assert result.timings.total_ms >= 0.0
    assert vector_store.received_top_k == 2
    assert answer_generator.call_count == 0


def test_retrieve_rejects_empty_query() -> None:
    service = RAGService(
        embedding_service=FakeEmbeddingService(
            embedding=np.ones(3, dtype=np.float32),
        ),
        vector_store=FakeVectorStore(results=[]),
        prompt_builder=PromptBuilder(),
        answer_generator=FakeAnswerGenerator(answer="answer"),
    )

    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        service.retrieve(query="  ")


def test_ask_returns_no_context_without_calling_generator() -> None:
    embedding_service = FakeEmbeddingService(
        embedding=np.array(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    )

    vector_store = FakeVectorStore(
        results=[],
    )

    answer_generator = FakeAnswerGenerator(
        answer="should not be used",
    )

    service = RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        prompt_builder=PromptBuilder(),
        answer_generator=answer_generator,
    )

    result = service.ask(
        question="不存在的問題",
    )

    assert result.answer == RAGService.NO_CONTEXT_ANSWER
    assert result.retrieved_chunks == []
    assert answer_generator.call_count == 0


def test_ask_skips_generation_for_low_relevance_result() -> None:
    retrieved = make_retrieved_chunk()
    low_relevance_result = RetrievedChunk(
        chunk=retrieved.chunk,
        score=0.4,
    )
    answer_generator = FakeAnswerGenerator(
        answer="should not be used",
    )
    service = RAGService(
        embedding_service=FakeEmbeddingService(
            embedding=np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        ),
        vector_store=FakeVectorStore(
            results=[low_relevance_result],
        ),
        prompt_builder=PromptBuilder(),
        answer_generator=answer_generator,
        min_relevance_score=0.6,
    )

    result = service.ask("unrelated question")

    assert "找不到與問題足夠相關" in result.answer
    assert result.retrieved_chunks == [low_relevance_result]
    assert result.timings.prompt_build_ms == 0.0
    assert result.timings.generation_ms == 0.0
    assert result.generation_metadata == {}
    assert answer_generator.call_count == 0


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "\n",
    ],
)
def test_ask_rejects_empty_question(
    question: str,
) -> None:
    service = RAGService(
        embedding_service=FakeEmbeddingService(
            embedding=np.ones(
                3,
                dtype=np.float32,
            ),
        ),
        vector_store=FakeVectorStore(
            results=[],
        ),
        prompt_builder=PromptBuilder(),
        answer_generator=FakeAnswerGenerator(
            answer="answer",
        ),
    )

    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        service.ask(
            question=question,
        )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_ask_rejects_invalid_top_k(
    top_k: int,
) -> None:
    service = RAGService(
        embedding_service=FakeEmbeddingService(
            embedding=np.ones(
                3,
                dtype=np.float32,
            ),
        ),
        vector_store=FakeVectorStore(
            results=[],
        ),
        prompt_builder=PromptBuilder(),
        answer_generator=FakeAnswerGenerator(
            answer="answer",
        ),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be greater than 0",
    ):
        service.ask(
            question="question",
            top_k=top_k,
        )


def test_ask_rejects_empty_generated_answer() -> None:
    service = RAGService(
        embedding_service=FakeEmbeddingService(
            embedding=np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        ),
        vector_store=FakeVectorStore(
            results=[
                make_retrieved_chunk(),
            ],
        ),
        prompt_builder=PromptBuilder(),
        answer_generator=FakeAnswerGenerator(
            answer="   ",
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="answer generator returned an empty answer",
    ):
        service.ask(
            question="Transformer 是什麼？",
        )
