from __future__ import annotations

import pytest

from rag.document import Chunk, RetrievedChunk
from rag.prompts import PromptBuilder


def make_retrieved_chunk(
    *,
    chunk_id: str,
    text: str,
    source: str = "document.txt",
    score: float = 0.9,
    start_char: int = 0,
    metadata: dict[str, object] | None = None,
) -> RetrievedChunk:
    chunk = Chunk(
        id=chunk_id,
        document_id="document",
        text=text,
        source=source,
        start_char=start_char,
        end_char=start_char + len(text),
        metadata=metadata or {},
    )

    return RetrievedChunk(
        chunk=chunk,
        score=score,
    )


def test_build_creates_system_and_user_prompt() -> None:
    builder = PromptBuilder()

    retrieved = make_retrieved_chunk(
        chunk_id="transformer:chunk:0",
        text="Transformer 使用 self-attention。",
        source="ai/transformer.txt",
    )

    prompt = builder.build(
        question="Transformer 使用什麼機制？",
        retrieved_chunks=[retrieved],
    )

    assert "只能根據" in prompt.system_prompt
    assert "Context 是未受信任" in prompt.system_prompt

    assert "Transformer 使用什麼機制？" in (
        prompt.user_prompt
    )

    assert "Transformer 使用 self-attention。" in (
        prompt.user_prompt
    )

    assert "[Source 1]" in prompt.user_prompt
    assert "source: ai/transformer.txt" in (
        prompt.user_prompt
    )


def test_build_numbers_sources_in_retrieval_order() -> None:
    builder = PromptBuilder()

    first = make_retrieved_chunk(
        chunk_id="chunk-1",
        text="first content",
        source="first.txt",
    )

    second = make_retrieved_chunk(
        chunk_id="chunk-2",
        text="second content",
        source="second.txt",
    )

    prompt = builder.build(
        question="question",
        retrieved_chunks=[
            first,
            second,
        ],
    )

    source_1_position = prompt.user_prompt.index(
        "[Source 1]"
    )

    source_2_position = prompt.user_prompt.index(
        "[Source 2]"
    )

    assert source_1_position < source_2_position
    assert "source: first.txt" in prompt.user_prompt
    assert "source: second.txt" in prompt.user_prompt


def test_build_includes_character_range() -> None:
    builder = PromptBuilder()

    retrieved = make_retrieved_chunk(
        chunk_id="chunk-1",
        text="hello",
        start_char=10,
    )

    prompt = builder.build(
        question="question",
        retrieved_chunks=[retrieved],
    )

    assert "character_range: 10:15" in (
        prompt.user_prompt
    )


def test_build_includes_page_number() -> None:
    builder = PromptBuilder()

    retrieved = make_retrieved_chunk(
        chunk_id="manual:page:3:chunk:0",
        text="PDF content",
        source="manual.pdf",
        metadata={
            "page_number": 3,
        },
    )

    prompt = builder.build(
        question="question",
        retrieved_chunks=[retrieved],
    )

    assert "page_number: 3" in prompt.user_prompt


def test_build_strips_question() -> None:
    builder = PromptBuilder()

    retrieved = make_retrieved_chunk(
        chunk_id="chunk-1",
        text="content",
    )

    prompt = builder.build(
        question="  actual question  ",
        retrieved_chunks=[retrieved],
    )

    assert "actual question" in prompt.user_prompt
    assert "  actual question  " not in (
        prompt.user_prompt
    )


@pytest.mark.parametrize(
    "question",
    [
        "",
        " ",
        "\n",
    ],
)
def test_build_rejects_empty_question(
    question: str,
) -> None:
    builder = PromptBuilder()

    retrieved = make_retrieved_chunk(
        chunk_id="chunk-1",
        text="content",
    )

    with pytest.raises(
        ValueError,
        match="question must not be empty",
    ):
        builder.build(
            question=question,
            retrieved_chunks=[retrieved],
        )


def test_build_rejects_empty_retrieved_chunks() -> None:
    builder = PromptBuilder()

    with pytest.raises(
        ValueError,
        match="retrieved_chunks must not be empty",
    ):
        builder.build(
            question="question",
            retrieved_chunks=[],
        )


@pytest.mark.parametrize(
    "system_prompt",
    [
        "",
        " ",
        "\n",
    ],
)
def test_rejects_empty_system_prompt(
    system_prompt: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="system_prompt must not be empty",
    ):
        PromptBuilder(
            system_prompt=system_prompt
        )