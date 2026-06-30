from dataclasses import dataclass, field
import uuid

from app.parser import PDFParser


@dataclass
class TextChunk:
    chunk_id: str
    document_id: str
    text_content: str
    chunk_index: int
    token_count: int
    page_number: int
    embedding_vector: list = field(default_factory=list)


class FixedSizeChunker:
    chunk_size: int = 400
    overlap: int = 50
    MIN_WORDS: int = 20

    def chunk(
        self,
        text: str,
        page_number: int,
        document_id: str,
        start_index: int
    ) -> list[TextChunk]:

        words = text.split()

        if len(words) < self.MIN_WORDS:
            return []

        chunks = []
        index = start_index
        step = self.chunk_size - self.overlap

        for start in range(0, len(words), step):
            chunk_words = words[start:start + self.chunk_size]

            if len(chunk_words) < self.MIN_WORDS:
                continue

            chunk_text = " ".join(chunk_words)

            chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    text_content=chunk_text,
                    chunk_index=index,
                    token_count=len(chunk_words),
                    page_number=page_number,
                    embedding_vector=[]
                )
            )

            index += 1

        return chunks


class RecursiveChunker:
    chunk_size: int = 400
    overlap: int = 50
    SEPARATORS: list[str] = ["\n\n", "\n", ". ", " "]
    MIN_WORDS: int = 20

    def chunk(
        self,
        text: str,
        page_number: int,
        document_id: str,
        start_index: int
    ) -> list[TextChunk]:

        segments = self._split_text(text, self.SEPARATORS)
        merged_segments = self._merge_with_overlap(segments)

        chunks = []
        index = start_index

        for segment in merged_segments:
            words = segment.split()

            if len(words) < self.MIN_WORDS:
                continue

            chunks.append(
                TextChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    text_content=segment,
                    chunk_index=index,
                    token_count=len(words),
                    page_number=page_number,
                    embedding_vector=[]
                )
            )

            index += 1

        if not chunks:
            return self._hard_split(text, page_number, document_id, start_index)

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
        current_words = []

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
        start_index: int
    ) -> list[TextChunk]:

        fixed_chunker = FixedSizeChunker()
        return fixed_chunker.chunk(
            text=text,
            page_number=page_number,
            document_id=document_id,
            start_index=start_index
        )


def parseAndChunk(
    file_path: str,
    chunker: FixedSizeChunker | RecursiveChunker | None = None
) -> list[TextChunk]:

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
            start_index=global_chunk_index
        )

        all_chunks.extend(page_chunks)
        global_chunk_index += len(page_chunks)

    if not all_chunks:
        raise ValueError("No chunks could be created from the document.")

    return all_chunks