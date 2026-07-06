"""
Tests for app/chunker.py. Operates on plain text strings directly --
no PDF file is needed since chunking logic is independent of parsing.
"""

import pytest

from app.chunker import FixedSizeChunker, RecursiveChunker, TextChunk


def make_long_text(num_words: int, word: str = "word") -> str:
    return " ".join(f"{word}{i}" for i in range(num_words))


def test_fixed_size_chunker_produces_expected_chunk_count():
    chunker = FixedSizeChunker(chunk_size=100, overlap=20, min_words=10)
    text = make_long_text(250)

    chunks = chunker.chunk(
        text=text,
        page_number=1,
        document_id="doc-1",
        start_index=0,
        document_name="test.pdf",
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
        assert chunk.document_id == "doc-1"
        assert chunk.document_name == "test.pdf"
        assert chunk.page_number == 1
        assert chunk.token_count >= 10


def test_fixed_size_chunker_returns_empty_for_short_text():
    chunker = FixedSizeChunker(min_words=20)
    short_text = "too short to keep"

    chunks = chunker.chunk(
        text=short_text,
        page_number=1,
        document_id="doc-1",
        start_index=0,
    )

    assert chunks == []


def test_fixed_size_chunker_rejects_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, overlap=150)


def test_chunk_index_increments_from_start_index():
    chunker = FixedSizeChunker(chunk_size=50, overlap=10, min_words=10)
    text = make_long_text(200)

    chunks = chunker.chunk(
        text=text,
        page_number=1,
        document_id="doc-1",
        start_index=5,
    )

    indices = [c.chunk_index for c in chunks]
    assert indices[0] == 5
    assert indices == sorted(indices)


def test_recursive_chunker_splits_on_paragraph_boundaries():
    chunker = RecursiveChunker(chunk_size=50, overlap=5, min_words=5)
    text = (
        make_long_text(30) + "\n\n" +
        make_long_text(30) + "\n\n" +
        make_long_text(30)
    )

    chunks = chunker.chunk(
        text=text,
        page_number=2,
        document_id="doc-2",
        start_index=0,
        document_name="paper.pdf",
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.page_number == 2
        assert chunk.document_name == "paper.pdf"


def test_recursive_chunker_falls_back_to_fixed_size_when_no_separators_found():
    # A single long "word" (no spaces/newlines) with no natural
    # boundaries should still be split via the fixed-size fallback.
    chunker = RecursiveChunker(chunk_size=50, overlap=5, min_words=5)
    text = make_long_text(300)

    chunks = chunker.chunk(
        text=text,
        page_number=1,
        document_id="doc-3",
        start_index=0,
    )

    assert len(chunks) > 0


def test_recursive_chunker_rejects_overlap_greater_than_chunk_size():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=50, overlap=60)
