from pathlib import Path

from rag.document import Document
from rag.loaders.base import FileLoaderStrategy


class TextFileLoader(FileLoaderStrategy):
    supported_suffixes = frozenset({
        ".txt",
        ".md",
        ".markdown",
    })

    def load(
        self,
        path: Path,
        relative_path: Path,
    ) -> list[Document]:
        text = path.read_text(encoding="utf-8")

        suffix = path.suffix.lower()

        if suffix == ".txt":
            file_type = "text"
        else:
            file_type = "markdown"

        return [
            Document(
                id=relative_path.with_suffix("").as_posix(),
                text=text,
                source=relative_path.as_posix(),
                metadata={
                    "file_type": file_type,
                    "file_name": path.name,
                    "suffix": suffix,
                },
            )
        ]