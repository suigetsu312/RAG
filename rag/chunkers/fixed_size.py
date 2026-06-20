from __future__ import annotations
from rag.chunkers.base import Chunker
from rag.document import Chunk, Document


class FixedSizeChunker(Chunker):
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        if not document.text:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap

        for chunk_index, start_char in enumerate(
            range(0, len(document.text), step)
        ):
            end_char = min(
                start_char + self.chunk_size,
                len(document.text),
            )

            chunk_text = document.text[start_char:end_char]

            if not chunk_text:
                break

            chunks.append(
                Chunk(
                    id=f"{document.id}:chunk:{chunk_index}",
                    document_id=document.id,
                    text=chunk_text,
                    source=document.source,
                    start_char=start_char,
                    end_char=end_char,
                    metadata=dict(document.metadata),
                )
            )

            if end_char >= len(document.text):
                break

        return chunks