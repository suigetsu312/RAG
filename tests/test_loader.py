from pathlib import Path

import pytest

from rag.loaders.text_loader import TextDocumentLoader


def test_load_directory_reads_txt_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text(
        "document A",
        encoding="utf-8",
    )

    (tmp_path / "b.txt").write_text(
        "document B",
        encoding="utf-8",
    )

    loader = TextDocumentLoader()
    documents = loader.load_directory(tmp_path)

    assert len(documents) == 2

    assert documents[0].id == "a"
    assert documents[0].text == "document A"
    assert documents[0].source == "a.txt"

    assert documents[1].id == "b"
    assert documents[1].text == "document B"
    assert documents[1].source == "b.txt"


def test_load_directory_reads_nested_txt_files(
    tmp_path: Path,
) -> None:
    nested_directory = tmp_path / "ai"
    nested_directory.mkdir()

    (nested_directory / "transformer.txt").write_text(
        "self-attention",
        encoding="utf-8",
    )

    loader = TextDocumentLoader()
    documents = loader.load_directory(tmp_path)

    assert len(documents) == 1
    assert documents[0].id == "ai/transformer"
    assert documents[0].source == "ai/transformer.txt"
    assert documents[0].text == "self-attention"


def test_load_directory_ignores_non_txt_files(
    tmp_path: Path,
) -> None:
    (tmp_path / "document.txt").write_text(
        "included",
        encoding="utf-8",
    )

    (tmp_path / "config.json").write_text(
        '{"ignored": true}',
        encoding="utf-8",
    )

    loader = TextDocumentLoader()
    documents = loader.load_directory(tmp_path)

    assert len(documents) == 1
    assert documents[0].id == "document"


def test_load_directory_returns_sorted_documents(
    tmp_path: Path,
) -> None:
    (tmp_path / "c.txt").write_text("C", encoding="utf-8")
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")

    loader = TextDocumentLoader()
    documents = loader.load_directory(tmp_path)

    assert [document.id for document in documents] == [
        "a",
        "b",
        "c",
    ]


def test_load_directory_raises_for_missing_directory(
    tmp_path: Path,
) -> None:
    missing_directory = tmp_path / "missing"

    loader = TextDocumentLoader()

    with pytest.raises(FileNotFoundError):
        loader.load_directory(missing_directory)


def test_load_directory_raises_when_path_is_file(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")

    loader = TextDocumentLoader()

    with pytest.raises(NotADirectoryError):
        loader.load_directory(file_path)