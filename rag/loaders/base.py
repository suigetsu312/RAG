from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag.document import Document


class DocumentLoader(Protocol):
    def load_file(
        self,
        path: str | Path,
    ) -> list[Document]:
        ...

    def load_directory(
        self,
        directory: str | Path,
    ) -> list[Document]:
        ...


class FileLoaderStrategy(Protocol):
    supported_suffixes: frozenset[str]

    def load(
        self,
        path: Path,
        relative_path: Path,
    ) -> list[Document]:
        ...
