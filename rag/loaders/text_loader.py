from __future__ import annotations

from pathlib import Path

from rag.document import Document


class TextDocumentLoader:
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

        for path in sorted(root.rglob("*.txt")):
            relative_path = path.relative_to(root)
            text = path.read_text(encoding="utf-8")

            document = Document(
                id=relative_path.with_suffix("").as_posix(),
                text=text,
                source=relative_path.as_posix(),
                metadata={
                    "file_type": "text",
                    "file_name": path.name,
                    "suffix": path.suffix.lower(),
                },
            )

            documents.append(document)

        return documents