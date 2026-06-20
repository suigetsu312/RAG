from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag.document import Document


class DocumentLoader(Protocol):
    def load_directory(
        self,
        directory: str | Path,
    ) -> list[Document]:
        ...