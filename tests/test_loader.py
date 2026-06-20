from pathlib import Path

import pymupdf
import pytest

from rag.loaders import (
    MultiFormatDocumentLoader,
    PDFFileLoader,
    TextFileLoader,
)


def make_loader() -> MultiFormatDocumentLoader:
    return MultiFormatDocumentLoader(
        strategies=[
            TextFileLoader(),
            PDFFileLoader(),
        ]
    )


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

    loader = make_loader()
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

    loader = make_loader()
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

    loader = make_loader()
    documents = loader.load_directory(tmp_path)

    assert len(documents) == 1
    assert documents[0].id == "document"


def test_load_directory_returns_sorted_documents(
    tmp_path: Path,
) -> None:
    (tmp_path / "c.txt").write_text("C", encoding="utf-8")
    (tmp_path / "a.txt").write_text("A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("B", encoding="utf-8")

    loader = make_loader()
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

    loader = make_loader()

    with pytest.raises(FileNotFoundError):
        loader.load_directory(missing_directory)


def test_load_directory_raises_when_path_is_file(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "document.txt"
    file_path.write_text("content", encoding="utf-8")

    loader = make_loader()

    with pytest.raises(NotADirectoryError):
        loader.load_directory(file_path)


def test_load_file_reads_markdown_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Notes", encoding="utf-8")

    documents = make_loader().load_file(path)

    assert len(documents) == 1
    assert documents[0].text == "# Notes"
    assert documents[0].metadata["file_type"] == "markdown"


def test_load_pdf_creates_one_document_per_text_page(
    tmp_path: Path,
) -> None:
    path = tmp_path / "paper.pdf"

    with pymupdf.open() as pdf:
        first_page = pdf.new_page()
        first_page.insert_text((72, 72), "First page")
        pdf.new_page()
        third_page = pdf.new_page()
        third_page.insert_text((72, 72), "Third page")
        pdf.save(path)

    documents = make_loader().load_file(path)

    assert [document.id for document in documents] == [
        "paper:page:1",
        "paper:page:3",
    ]
    assert documents[0].text == "First page"
    assert documents[1].text == "Third page"
    assert documents[0].metadata["file_type"] == "pdf"
    assert documents[0].metadata["page_count"] == 3


def test_load_file_rejects_unsupported_type(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Unsupported document file type",
    ):
        make_loader().load_file(path)
