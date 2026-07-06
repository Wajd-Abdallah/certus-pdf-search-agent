"""
Tests for app/indexer.py. Uses a temporary, isolated ChromaDB directory
(via pytest's tmp_path) so tests never touch the live app's or the
evaluation's persistent storage, and are automatically cleaned up.
"""

import uuid

import pytest

from app.chunker import TextChunk
from app.indexer import Indexer


def make_chunk(text: str, document_name: str = "test.pdf", page: int = 1) -> TextChunk:
    return TextChunk(
        chunk_id=str(uuid.uuid4()),
        document_id=str(uuid.uuid4()),
        document_name=document_name,
        text_content=text,
        chunk_index=0,
        token_count=len(text.split()),
        page_number=page,
    )


@pytest.fixture
def indexer(tmp_path):
    return Indexer(
        collection_name="test_collection",
        persist_directory=str(tmp_path / "chroma_test_db"),
    )


def test_index_and_search_returns_relevant_chunk(indexer):
    chunk = make_chunk("The quick brown fox jumps over the lazy dog.")
    indexer.index_chunks([chunk])

    results = indexer.search("quick brown fox", top_k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == chunk.chunk_id
    assert results[0]["metadata"]["document_name"] == "test.pdf"
    assert results[0]["metadata"]["page_number"] == 1
    assert "distance" in results[0]


def test_index_chunks_raises_on_empty_list(indexer):
    with pytest.raises(ValueError):
        indexer.index_chunks([])


def test_search_raises_on_empty_query(indexer):
    with pytest.raises(ValueError):
        indexer.search("", top_k=5)


def test_search_returns_empty_list_when_collection_is_empty(indexer):
    results = indexer.search("anything", top_k=5)
    assert results == []


def test_clear_removes_previously_indexed_chunks(indexer):
    chunk = make_chunk("Some content to be cleared later.")
    indexer.index_chunks([chunk])

    indexer.clear()
    results = indexer.search("content", top_k=5)

    assert results == []


def test_search_respects_top_k(indexer):
    chunks = [make_chunk(f"Document number {i} about cats and dogs.") for i in range(5)]
    indexer.index_chunks(chunks)

    results = indexer.search("cats and dogs", top_k=3)

    assert len(results) == 3
