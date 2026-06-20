from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    file_name: str
    source: str
    sha256: str
    size_bytes: int
    document_count: int
    chunk_count: int
    created_at: str


class DocumentManifest:
    FORMAT_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        records: list[DocumentRecord] | None = None,
    ) -> None:
        self._path = Path(path)
        self._records = {
            record.id: record
            for record in records or []
        }
        self._lock = RLock()

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> DocumentManifest:
        manifest_path = Path(path)

        if not manifest_path.exists():
            return cls(manifest_path)

        payload = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        version = payload.get("version")

        if version != cls.FORMAT_VERSION:
            raise RuntimeError(
                "Unsupported document manifest version: "
                f"{version}"
            )

        records = [
            DocumentRecord(**item)
            for item in payload.get("documents", [])
        ]

        return cls(
            path=manifest_path,
            records=records,
        )

    def list_documents(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(
                self._records.values(),
                key=lambda record: record.created_at,
                reverse=True,
            )

    def find_by_id(
        self,
        document_id: str,
    ) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(document_id)

    def find_by_sha256(
        self,
        sha256: str,
    ) -> DocumentRecord | None:
        with self._lock:
            for record in self._records.values():
                if record.sha256 == sha256:
                    return record

        return None

    def find_by_file_name(
        self,
        file_name: str,
    ) -> DocumentRecord | None:
        with self._lock:
            for record in self._records.values():
                if record.file_name == file_name:
                    return record

        return None

    def add(
        self,
        record: DocumentRecord,
    ) -> None:
        with self._lock:
            if record.id in self._records:
                raise ValueError(
                    f"Document already exists: {record.id}"
                )

            if self.find_by_sha256(record.sha256):
                raise ValueError(
                    "Document with the same content already exists"
                )

            if self.find_by_file_name(record.file_name):
                raise ValueError(
                    "Document with the same filename already exists"
                )

            self._records[record.id] = record
            self._save_locked()

    def _save_locked(self) -> None:
        self._path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "version": self.FORMAT_VERSION,
            "documents": [
                asdict(record)
                for record in self._records.values()
            ],
        }

        temporary_path = self._path.with_suffix(
            f"{self._path.suffix}.tmp"
        )

        temporary_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        os.replace(
            temporary_path,
            self._path,
        )

    def replace_all(
        self,
        records: list[DocumentRecord],
    ) -> None:
        with self._lock:
            self._records = {
                record.id: record
                for record in records
            }
            self._save_locked()

    def source_hashes(self) -> dict[str, str]:
        with self._lock:
            return {
                record.source: record.sha256
                for record in self._records.values()
            }


def calculate_sha256(
    path: str | Path,
    block_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)

    return digest.hexdigest()
