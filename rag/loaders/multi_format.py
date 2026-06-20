from __future__ import annotations

from pathlib import Path

from rag.document import Document
from rag.loaders.base import (
    DocumentLoader,
    FileLoaderStrategy,
)


class MultiFormatDocumentLoader(DocumentLoader):
    def __init__(
        self,
        strategies: list[FileLoaderStrategy],
    ) -> None:
        if not strategies:
            raise ValueError(
                "strategies must not be empty"
            )

        self._strategies: dict[str, FileLoaderStrategy] = {}

        for strategy in strategies:
            for suffix in strategy.supported_suffixes:
                normalized_suffix = suffix.lower()

                if not normalized_suffix.startswith("."):
                    raise ValueError(
                        "supported suffix must start with '.': "
                        f"{suffix}"
                    )

                if normalized_suffix in self._strategies:
                    raise ValueError(
                        "duplicate loader strategy for suffix: "
                        f"{normalized_suffix}"
                    )

                self._strategies[normalized_suffix] = strategy

    def load_file(
        self,
        path: str | Path,
    ) -> list[Document]:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Document file does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise IsADirectoryError(
                f"Document path is not a file: {file_path}"
            )

        strategy = self._strategies.get(
            file_path.suffix.lower()
        )

        if strategy is None:
            raise ValueError(
                "Unsupported document file type: "
                f"{file_path.suffix}"
            )

        return strategy.load(
            path=file_path,
            relative_path=Path(file_path.name),
        )

    def load_directory(
        self,
        directory: str | Path,
    ) -> list[Document]:
        root = Path(directory)

        if not root.exists():
            raise FileNotFoundError(
                f"Document directory does not exist: {root}"
            )

        if not root.is_dir():
            raise NotADirectoryError(
                f"Document path is not a directory: {root}"
            )

        documents: list[Document] = []

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue

            strategy = self._strategies.get(
                path.suffix.lower()
            )

            if strategy is None:
                continue

            relative_path = path.relative_to(root)

            documents.extend(
                strategy.load(
                    path=path,
                    relative_path=relative_path,
                )
            )

        return documents