import pytest

from rag.chunkers import FixedSizeChunker, RoutingChunker
from rag.document import Document


def test_split_document_with_overlap() -> None:
    document = Document(
        id="doc-1",
        text="0123456789",
        source="memory",
    )

    chunker = FixedSizeChunker(
        chunk_size=6,
        chunk_overlap=2,
    )

    chunks = chunker.split(document)

    assert len(chunks) == 2

    assert chunks[0].id == "doc-1:chunk:0"
    assert chunks[0].text == "012345"
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 6

    assert chunks[1].id == "doc-1:chunk:1"
    assert chunks[1].text == "456789"
    assert chunks[1].start_char == 4
    assert chunks[1].end_char == 10


def test_chunk_offsets_match_original_document() -> None:
    document = Document(
        id="doc-1",
        text="abcdefghijklmnopqrstuvwxyz",
        source="memory",
    )

    chunker = FixedSizeChunker(
        chunk_size=10,
        chunk_overlap=3,
    )

    chunks = chunker.split(document)

    for chunk in chunks:
        extracted_text = document.text[
            chunk.start_char:chunk.end_char
        ]

        assert extracted_text == chunk.text


def test_short_document_produces_one_chunk() -> None:
    document = Document(
        id="doc-1",
        text="short",
        source="memory",
    )

    chunker = FixedSizeChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.split(document)

    assert len(chunks) == 1
    assert chunks[0].text == "short"
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 5


def test_empty_document_produces_no_chunks() -> None:
    document = Document(
        id="doc-1",
        text="",
        source="memory",
    )

    chunker = FixedSizeChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.split(document)

    assert chunks == []


def test_chunk_preserves_document_information() -> None:
    document = Document(
        id="ai/transformer",
        text="Transformer uses self-attention.",
        source="ai/transformer.txt",
        metadata={
            "file_type": "text",
            "language": "en",
        },
    )

    chunker = FixedSizeChunker(
        chunk_size=20,
        chunk_overlap=5,
    )

    chunks = chunker.split(document)

    assert chunks[0].document_id == document.id
    assert chunks[0].source == document.source
    assert chunks[0].metadata == document.metadata
    assert chunks[0].metadata is not document.metadata


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (10, -1),
        (10, 10),
        (10, 11),
    ],
)
def test_invalid_configuration_raises(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        FixedSizeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_routing_chunker_selects_chunker_by_file_type() -> None:
    chunker = RoutingChunker(
        chunkers={
            "pdf": FixedSizeChunker(
                chunk_size=10,
                chunk_overlap=0,
            ),
        },
        default_chunker=FixedSizeChunker(
            chunk_size=6,
            chunk_overlap=0,
        ),
    )

    pdf_chunks = chunker.split(
        Document(
            id="paper:page:1",
            text="0123456789",
            source="paper.pdf",
            metadata={"file_type": "pdf"},
        )
    )
    text_chunks = chunker.split(
        Document(
            id="notes",
            text="0123456789",
            source="notes.txt",
            metadata={"file_type": "text"},
        )
    )

    assert len(pdf_chunks) == 1
    assert len(text_chunks) == 2
