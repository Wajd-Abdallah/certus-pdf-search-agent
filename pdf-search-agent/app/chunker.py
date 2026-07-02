"""
Splits parsed PDF page text into TextChunk objects ready for embedding.
Depends on app/parser.py (PDFParser) for the (DocumentData, {page: text}) input.
"""

from dataclasses import dataclass, field
import uuid

from app.parser import PDFParser


# ==========================================================
# TextChunk
# ==========================================================
@dataclass
class TextChunk:
    chunk_id: str
    document_id: str
    document_name: str  # original filename, needed for citations later
    text_content: str
    chunk_index: int
    token_count: int  # NOTE: this is a WORD count approximation, not a real tokenizer count
    page_number: int
    embedding_vector: list = field(default_factory=list)


# ==========================================================
# FixedSizeChunker
# ==========================================================
class FixedSizeChunker:
    """
    Splits text into fixed-size overlapping windows, measured in words.
    """

    def __init__(self, chunk_size: int = 400, overlap: int = 50, min_words: int = 20):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_words = min_words

    def chunk(
        self,
        text: str,
        page_number: int,
        document_id: str,
        start_index: int,
        document_name: str = "",
    ) -> list[TextChunk]:

        words = text.split()

        if len(words) < self.min_words:
            return []

        chunks = []
        index = start_index
        step = self.chunk_size - self.overlap  # always > 0, validated in __init__

        for start in range(0, len(words), step):
            chunk_words = words[start:start + self.chunk_size]

            if len(chunk_words) < self.min_words:
                continue

            chunk_text = " ".join(chunk_words)

            chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    document_name=document_name,
                    text_content=chunk_text,
                    chunk_index=index,
                    token_count=len(chunk_words),
                    page_number=page_number,
                    embedding_vector=[]
                )
            )

            index += 1

        return chunks


# ==========================================================
# RecursiveChunker
# ==========================================================
class RecursiveChunker:
    """
    Splits text on natural boundaries (paragraph -> line -> sentence -> word)
    before falling back to fixed-size chunking.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(
        self,
        chunk_size: int = 400,
        overlap: int = 50,
        min_words: int = 20,
        separators: list[str] | None = None,
    ):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_words = min_words
        self.separators = separators if separators is not None else list(self.DEFAULT_SEPARATORS)

    def chunk(
        self,
        text: str,
        page_number: int,
        document_id: str,
        start_index: int,
        document_name: str = "",
    ) -> list[TextChunk]:

        segments = self._split_text(text, self.separators)
        merged_segments = self._merge_with_overlap(segments)

        chunks = []
        index = start_index

        for segment in merged_segments:
            words = segment.split()

            if len(words) < self.min_words:
                continue

            chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    document_name=document_name,
                    text_content=segment,
                    chunk_index=index,
                    token_count=len(words),
                    page_number=page_number,
                    embedding_vector=[]
                )
            )

            index += 1

        if not chunks:
            return self._hard_split(text, page_number, document_id, start_index, document_name)

        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]

        separator = separators[0]

        if separator not in text:
            return self._split_text(text, separators[1:])

        parts = text.split(separator)

        segments = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            if len(part.split()) > self.chunk_size:
                segments.extend(self._split_text(part, separators[1:]))
            else:
                segments.append(part)

        return segments

    def _merge_with_overlap(self, segments: list[str]) -> list[str]:
        merged = []
        current_words: list[str] = []

        for segment in segments:
            words = segment.split()

            if len(current_words) + len(words) <= self.chunk_size:
                current_words.extend(words)
            else:
                if current_words:
                    merged.append(" ".join(current_words))

                overlap_words = current_words[-self.overlap:] if self.overlap > 0 else []
                current_words = overlap_words + words

        if current_words:
            merged.append(" ".join(current_words))

        return merged

    def _hard_split(
        self,
        text: str,
        page_number: int,
        document_id: str,
        start_index: int,
        document_name: str = "",
    ) -> list[TextChunk]:
        fixed_chunker = FixedSizeChunker(
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            min_words=self.min_words,
        )
        return fixed_chunker.chunk(
            text=text,
            page_number=page_number,
            document_id=document_id,
            start_index=start_index,
            document_name=document_name,
        )


# ==========================================================
# parse_and_chunk
# ==========================================================
def parse_and_chunk(
    file_path: str,
    chunker: FixedSizeChunker | RecursiveChunker | None = None
) -> list[TextChunk]:
    """
    Parses one PDF and splits it into TextChunks.
    """
    if chunker is None:
        chunker = RecursiveChunker()

    parser = PDFParser()
    document_data, page_texts = parser.parse(file_path)

    all_chunks = []
    global_chunk_index = 0

    for page_number, text in page_texts.items():
        page_chunks = chunker.chunk(
            text=text,
            page_number=page_number,
            document_id=document_data.document_id,
            start_index=global_chunk_index,
            document_name=document_data.file_name,
        )

        all_chunks.extend(page_chunks)
        global_chunk_index += len(page_chunks)

    if not all_chunks:
        raise ValueError("No chunks could be created from the document.")

    return all_chunks


# Backward-compatible alias in case other files still import the old camelCase name.
# TODO: remove once all callers use parse_and_chunk().
parseAndChunk = parse_and_chunk