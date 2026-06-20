from __future__ import annotations

from typing import Protocol

from rag.document import Chunk, Document


class Chunker(Protocol):
    def split(
        self,
        document: Document,
    ) -> list[Chunk]:
        ...