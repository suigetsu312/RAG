from pathlib import Path

import pymupdf

from rag.document import Document
from rag.loaders.base import FileLoaderStrategy


class PDFFileLoader(FileLoaderStrategy):
    supported_suffixes = frozenset({
        ".pdf",
    })

    def load(
        self,
        path: Path,
        relative_path: Path,
    ) -> list[Document]:
        documents: list[Document] = []
        base_id = relative_path.with_suffix("").as_posix()

        with pymupdf.open(path) as pdf:
            page_count = len(pdf)

            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()

                if not text:
                    continue

                page_number = page_index + 1

                documents.append(
                    Document(
                        id=f"{base_id}:page:{page_number}",
                        text=text,
                        source=relative_path.as_posix(),
                        metadata={
                            "file_type": "pdf",
                            "file_name": path.name,
                            "suffix": ".pdf",
                            "page_number": page_number,
                            "page_count": page_count,
                        },
                    )
                )

        return documents