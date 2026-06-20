from rag.chunkers.base import Chunker
from rag.document import Chunk, Document


class RoutingChunker(Chunker):
    def __init__(
        self,
        chunkers: dict[str, Chunker],
        default_chunker: Chunker,
    ) -> None:
        self._chunkers = chunkers
        self._default_chunker = default_chunker

    def split(
        self,
        document: Document,
    ) -> list[Chunk]:
        file_type = str(
            document.metadata.get("file_type", "")
        )

        chunker = self._chunkers.get(
            file_type,
            self._default_chunker,
        )

        return chunker.split(document)