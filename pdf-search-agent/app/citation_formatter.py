"""
Builds structured Citation objects from an LLM-generated answer's inline
citation markers (e.g. "[document.pdf:p3]" or "[document.pdf:3]"),
validated against the actually retrieved chunks.

Depends on app/retriever.py's output shape:
    {"chunk_id", "text_content", "metadata": {..., "document_name", "page_number"}, "distance"}

Replaces the earlier approach of attaching a citation for every retrieved
chunk regardless of whether the generated answer actually used it.
"""

import re

from app.schemas import Citation

# Tolerant of both "[doc:p15]" and "[doc:15]" -- local LLMs don't always
# reproduce the exact instructed format, so the "p" is treated as optional
# rather than relying on perfect compliance.
CITATION_PATTERN = re.compile(r"\[([^\]:]+):p?(\d+)\]")


def extract_inline_citations(answer_text: str) -> list[dict]:
    """
    Scans generated answer text for inline citation markers and returns
    them as a list of {"document": ..., "page_number": ...} dicts, in
    the order they appear (duplicates included -- deduplication happens
    in format_citations).
    """
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
    """
    Builds the final citation list by:
    1. Extracting the inline markers the model actually wrote in its answer.
    2. Deduplicating on (document, page) -- multiple claims citing the
       same page produce one citation badge, not several.
    3. Validating each citation against the retrieved chunks, so a
       citation is only kept if that document/page was genuinely
       retrieved (protects against the model inventing a page number).
    """
    retrieved_lookup = {
        (chunk.get("metadata", {}).get("document_name"), chunk.get("metadata", {}).get("page_number")): chunk
        for chunk in chunks
    }

    inline_citations = extract_inline_citations(answer_text)

    citations = []
    seen = set()

    for citation in inline_citations:
        key = (citation["document"], citation["page_number"])
        if key in seen:
            continue

        matching_chunk = retrieved_lookup.get(key)
        if matching_chunk is None:
            # The model cited a document/page that wasn't actually
            # retrieved -- skip it rather than presenting an
            # unverifiable citation to the user.
            continue

        seen.add(key)
        chunk_id = matching_chunk.get("chunk_id")
        citations.append(format_citation(citation["document"], citation["page_number"], chunk_id))

    return citations


def extract_contexts(chunks: list[dict]) -> list[str]:
    contexts = []
    for chunk in chunks:
        text = chunk.get("text_content") or ""
        if text.strip():
            contexts.append(text.strip())
    return contexts


# Backward-compatible aliases.
formatCitation = format_citation
extractContexts = extract_contexts