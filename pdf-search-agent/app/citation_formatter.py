"""
Builds structured Citation objects from an LLM-generated answer's inline
citation markers (e.g. "[document.pdf:p3]" or "[document.pdf:3]"),
validated against the actually retrieved chunks.
"""

import re

from app.schemas import Citation

CITATION_PATTERN = re.compile(r"\[([^\]:]+):p?(\d+)\]")


def _normalize_document_name(name: str) -> str:
    """
    Normalizes a document name for matching purposes only (not for
    display). Strips a trailing ".pdf" (case-insensitive) and
    lowercases, so citations still match correctly even if the LLM
    drops the file extension when writing an inline citation --
    a real, observed formatting inconsistency with local models.
    """
    name = name.strip().lower()
    if name.endswith(".pdf"):
        name = name[:-4]
    return name


def extract_inline_citations(answer_text: str) -> list[dict]:
    citations = []
    for match in CITATION_PATTERN.finditer(answer_text):
        document_name = match.group(1).strip()
        page_number = int(match.group(2))
        citations.append({"document": document_name, "page_number": page_number})
    return citations


def format_citation(source: str, page, chunk_id=None) -> dict:
    citation = Citation(
        document=source,
        page_number=page,
        chunk_id=chunk_id
    )
    return citation.toDict()


def format_citations(answer_text: str, chunks: list[dict]) -> list[dict]:
    # Keyed by NORMALIZED (document, page) for robust matching, but the
    # stored value keeps the real chunk so we can display the correct,
    # full filename regardless of what the model wrote.
    retrieved_lookup = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        doc_name = metadata.get("document_name")
        page = metadata.get("page_number")
        if doc_name is None or page is None:
            continue
        key = (_normalize_document_name(doc_name), page)
        retrieved_lookup[key] = chunk

    inline_citations = extract_inline_citations(answer_text)

    citations = []
    seen = set()

    for citation in inline_citations:
        key = (_normalize_document_name(citation["document"]), citation["page_number"])
        if key in seen:
            continue

        matching_chunk = retrieved_lookup.get(key)
        if matching_chunk is None:
            continue

        seen.add(key)
        real_metadata = matching_chunk.get("metadata", {})
        real_document_name = real_metadata.get("document_name")
        chunk_id = matching_chunk.get("chunk_id")
        citations.append(format_citation(real_document_name, citation["page_number"], chunk_id))

    return citations


def extract_contexts(chunks: list[dict]) -> list[str]:
    contexts = []
    for chunk in chunks:
        text = chunk.get("text_content") or ""
        if text.strip():
            contexts.append(text.strip())
    return contexts


formatCitation = format_citation
extractContexts = extract_contexts