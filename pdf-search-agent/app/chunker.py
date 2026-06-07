"""
chunker.py
==========
Text chunking strategies for the PDF Search Agent.

Responsibilities
----------------
- Split extracted PDF text into semantic chunks
- Attach metadata for retrieval/indexing
- Support multiple chunking strategies

Main output
-----------
parse_and_chunk(file_path) -> list[TextChunk]
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from parser import PDFParser



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@dataclass
class TextChunk:
    """
    A single text chunk extracted from a document.

    Attributes
    ----------
    chunk_id : str
        Unique identifier (UUID)

    document_id : str
        Parent document identifier

    text_content : str
        Chunk text

    chunk_index : int
        Sequential chunk index in document

    token_count : int
        Approximate token/word count

    page_number : int
        Source page number

    embedding_vector : list
        Filled later by indexer.py
    """

    chunk_id: str
    document_id: str
    text_content: str
    chunk_index: int
    token_count: int
    page_number: int
    embedding_vector: list = field(default_factory=list)



# Fixed Size Chunker

class FixedSizeChunker:
    """
    Splits text into fixed-size word windows.

    Parameters
    ----------
    chunk_size : int
        Maximum words per chunk

    overlap : int
        Number of overlapping words
    """

    MIN_WORDS = 20

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be positive."
            )

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be >= 0 and < chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

   
    def chunk(
        self,
        text: str,
        page_number: int,
        document_id: str,
        start_index: int = 0,
    ) -> list[TextChunk]:

        words = text.split()

        if not words:
            return []

        chunks: list[TextChunk] = []

        step = self.chunk_size - self.overlap

        chunk_index = start_index

        for i in range(0, len(words), step):

            window = words[i : i + self.chunk_size]

            if len(window) < self.MIN_WORDS:
                continue

            content = " ".join(window)

            chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    text_content=content,
                    chunk_index=chunk_index,
                    token_count=len(window),
                    page_number=page_number,
                )
            )

            chunk_index += 1

        return chunks



# Recursive Chunker


class RecursiveChunker:
    """
    Recursive semantic chunking.

    Splitting order:
        paragraphs -> lines -> sentences -> words

    More retrieval-friendly than fixed-size chunking.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " "]

    MIN_WORDS = 20

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> None:

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be positive."
            )

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be >= 0 and < chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

     

    def _hard_split(self, text: str) -> list[str]:
        """
        Final fallback split by word count.

        Ensures no chunk exceeds chunk_size.
        """

        words = text.split()

        return [
            " ".join(
                words[i : i + self.chunk_size]
            )
            for i in range(
                0,
                len(words),
                self.chunk_size,
            )
        ]

    

    def _split_text(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:

        if not separators:
            return self._hard_split(text)

        separator = separators[0]

        parts = [
            part.strip()
            for part in text.split(separator)
            if part.strip()
        ]

        results: list[str] = []

        for part in parts:

            word_count = len(part.split())

            if word_count <= self.chunk_size:
                results.append(part)

            else:
                results.extend(
                    self._split_text(
                        part,
                        separators[1:],
                    )
                )

        return results



    def _merge_with_overlap(
        self,
        segments: list[str],
    ) -> list[str]:

        chunks: list[str] = []

        current_words: list[str] = []

        for segment in segments:

            segment_words = segment.split()

            current_word_count = len(current_words)

            if (
                current_word_count
                + len(segment_words)
                <= self.chunk_size
            ):

                current_words.extend(segment_words)

            else:

                if current_words:

                    chunk_text = " ".join(current_words)

                    chunks.append(chunk_text)

                    overlap_words = current_words[
                        -self.overlap :
                    ]

                    current_words = (
                        overlap_words
                        + segment_words
                    )

                else:
                    current_words = segment_words

        if current_words:

            chunk_text = " ".join(current_words)

            chunks.append(chunk_text)

        return chunks


    def chunk(
        self,
        text: str,
        page_number: int,
        document_id: str,
        start_index: int = 0,
    ) -> list[TextChunk]:

        if not text.strip():
            return []

        segments = self._split_text(
            text,
            self.SEPARATORS,
        )

        merged_chunks = self._merge_with_overlap(
            segments
        )

        final_chunks: list[TextChunk] = []

        for i, content in enumerate(merged_chunks):

            word_count = len(content.split())

            if word_count < self.MIN_WORDS:
                continue

            final_chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    text_content=content,
                    chunk_index=start_index + i,
                    token_count=word_count,
                    page_number=page_number,
                )
            )

        return final_chunks


# Main pipeline function

def parseAndChunk(
    file_path: str,
    chunker: (
        FixedSizeChunker
        | RecursiveChunker
        | None
    ) = None,
) -> list[TextChunk]:
    """
    Parse a PDF and return all text chunks.

    Parameters
    
    file_path : str
        Path to the PDF file

    chunker : FixedSizeChunker | RecursiveChunker | None
        Chunking strategy

    Returns
    
    list[TextChunk]
        All generated chunks
    """

    if chunker is None:
        chunker = RecursiveChunker()

    parser = PDFParser()

    document, pages = parser.parse(file_path)

    all_chunks: list[TextChunk] = []

    global_index = 0

    for page_number in sorted(pages.keys()):

        text = pages[page_number]

        if not text.strip():
            continue

        try:

            page_chunks = chunker.chunk(
                text=text,
                page_number=page_number,
                document_id=document.document_id,
                start_index=global_index,
            )

            all_chunks.extend(page_chunks)

            global_index += len(page_chunks)

        except Exception as exc:

            logger.warning(
                "Could not chunk page %s: %s",
                page_number,
                exc,
            )

    logger.info(
        "Generated %s chunks from '%s'",
        len(all_chunks),
        document.file_name,
    )

    return all_chunks