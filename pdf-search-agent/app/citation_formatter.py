"""
Builds structured Citation objects from retrieved chunks.
Depends on app/retriever.py's output shape:
    {"chunk_id", "text_content", "metadata": {..., "document_name", "page_number"}, "distance"}
"""

from app.schemas import Citation


def format_citation(source: str, page, chunk_id=None) -> dict:
    citation = Citation(
        document=source,
        page_number=page,
        chunk_id=chunk_id
    )
    return citation.toDict()


def format_citations(chunks: list[dict]) -> list[dict]:
    citations = []
    seen = set()

    for index, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        source = metadata.get("document_name") or "Unknown"
        page = metadata.get("page_number", "?")
        chunk_id = chunk.get("chunk_id") or f"{source}_{page}_{index}"

        # Dedupe on document + page only: multiple chunks can come from the
        # same page, and we don't want one citation per chunk in that case.
        key = f"{source}_{page}"
        if key not in seen:
            seen.add(key)
            citations.append(format_citation(source, page, chunk_id))

    return citations


def extract_contexts(chunks: list[dict]) -> list[str]:
    contexts = []
    for chunk in chunks:
        text = chunk.get("text_content") or ""
        if text.strip():
            contexts.append(text.strip())
    return contexts


# Backward-compatible aliases in case other files still import the old camelCase names.
formatCitation = format_citation
formatCitations = format_citations
extractContexts = extract_contexts