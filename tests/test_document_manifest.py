from pathlib import Path

import pytest

from rag.document_manifest import (
    DocumentManifest,
    DocumentRecord,
    calculate_sha256,
)


def make_record(
    *,
    document_id: str = "hash-1",
    file_name: str = "paper.pdf",
    sha256: str = "hash-1",
) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        file_name=file_name,
        source=file_name,
        sha256=sha256,
        size_bytes=123,
        document_count=2,
        chunk_count=4,
        created_at="2026-06-20T00:00:00+00:00",
    )


def test_manifest_persists_and_loads_records(
    tmp_path: Path,
) -> None:
    path = tmp_path / "documents.json"
    manifest = DocumentManifest(path)
    record = make_record()

    manifest.add(record)
    loaded = DocumentManifest.load(path)

    assert loaded.find_by_id(record.id) == record
    assert loaded.find_by_sha256(record.sha256) == record
    assert loaded.find_by_file_name(record.file_name) == record
    assert loaded.source_hashes() == {
        record.source: record.sha256,
    }


def test_manifest_rejects_duplicate_content(
    tmp_path: Path,
) -> None:
    manifest = DocumentManifest(tmp_path / "documents.json")
    manifest.add(make_record())

    with pytest.raises(
        ValueError,
        match="same content",
    ):
        manifest.add(
            make_record(
                document_id="hash-2",
                file_name="copy.pdf",
                sha256="hash-1",
            )
        )


def test_calculate_sha256_uses_file_content(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("same content", encoding="utf-8")
    second.write_text("same content", encoding="utf-8")

    assert calculate_sha256(first) == calculate_sha256(second)
