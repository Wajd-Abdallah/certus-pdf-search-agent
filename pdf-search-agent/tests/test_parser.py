"""
Tests for app/parser.py's PDFParser. Uses a synthetically generated PDF
(via PyMuPDF itself) so no external file dependency is needed.
"""

import fitz
import pytest

from app.parser import PDFParser, DocumentData


def make_pdf(tmp_path, pages_text: list[str]) -> str:
    """Creates a small real PDF file with one page per string in pages_text."""
    path = tmp_path / "generated.pdf"
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
    return str(path)


def test_parse_returns_document_data_and_pages(tmp_path):
    pdf_path = make_pdf(tmp_path, ["Hello world", "Second page content"])

    parser = PDFParser()
    document, pages = parser.parse(pdf_path)

    assert isinstance(document, DocumentData)
    assert document.file_type == "application/pdf"
    assert document.num_pages == 2
    assert document.file_size > 0
    assert 1 in pages
    assert 2 in pages
    assert "Hello world" in pages[1]
    assert "Second page content" in pages[2]


def test_parse_raises_on_missing_file(tmp_path):
    parser = PDFParser()
    with pytest.raises(FileNotFoundError):
        parser.parse(str(tmp_path / "does_not_exist.pdf"))


def test_parse_raises_on_non_pdf_extension(tmp_path):
    fake = tmp_path / "note.txt"
    fake.write_text("not a pdf")

    parser = PDFParser()
    with pytest.raises(ValueError):
        parser.parse(str(fake))


def test_parse_raises_on_empty_file(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")

    parser = PDFParser()
    with pytest.raises(ValueError):
        parser.parse(str(empty))


def test_parse_raises_when_no_extractable_text(tmp_path):
    # A valid PDF with a blank page has no text to extract.
    path = tmp_path / "blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(path))
    doc.close()

    parser = PDFParser()
    with pytest.raises(ValueError):
        parser.parse(str(path))


def test_parse_multiple_skips_invalid_files_without_raising(tmp_path):
    good_pdf = make_pdf(tmp_path, ["Valid content here"])
    bad_txt = tmp_path / "bad.txt"
    bad_txt.write_text("not a pdf")

    parser = PDFParser()
    results = parser.parse_multiple([good_pdf, str(bad_txt)])

    assert len(results) == 1
    document, pages = results[0]
    assert "Valid content here" in pages[1]
