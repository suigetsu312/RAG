import pytest

from rag.document import Chunk, Document, RAGResult, RetrievedChunk


def test_document_can_be_created() -> None:
    document = Document(
        id="doc-1",
        text="hello world",
        source="memory",
    )

    assert document.id == "doc-1"
    assert document.text == "hello world"
    assert document.source == "memory"
    assert document.metadata == {}


def test_chunk_offsets_match_document_text() -> None:
    document = Document(
        id="doc-1",
        text="0123456789",
        source="memory",
    )

    chunk = Chunk(
        id="doc-1:chunk:0",
        document_id=document.id,
        text="2345",
        source=document.source,
        start_char=2,
        end_char=6,
    )

    assert document.text[chunk.start_char:chunk.end_char] == chunk.text


def test_retrieved_chunk_contains_chunk_and_score() -> None:
    chunk = Chunk(
        id="doc-1:chunk:0",
        document_id="doc-1",
        text="hello",
        source="memory",
        start_char=0,
        end_char=5,
    )

    retrieved = RetrievedChunk(
        chunk=chunk,
        score=0.9,
    )

    assert retrieved.chunk is chunk
    assert retrieved.score == pytest.approx(0.9)


def test_rag_result_contains_answer_and_sources() -> None:
    chunk = Chunk(
        id="doc-1:chunk:0",
        document_id="doc-1",
        text="hello",
        source="memory",
        start_char=0,
        end_char=5,
    )

    retrieved = RetrievedChunk(
        chunk=chunk,
        score=0.8,
    )

    result = RAGResult(
        answer="answer",
        retrieved_chunks=[retrieved],
    )

    assert result.answer == "answer"
    assert result.retrieved_chunks == [retrieved]