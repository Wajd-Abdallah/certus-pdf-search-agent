from app.schemas import Citation


def formatCitation(source: str, page, chunk_id=None) -> dict:
    citation = Citation(
        document=source,
        page_number=page,
        chunk_id=chunk_id
    )

    return citation.toDict()


def formatCitations(chunks: list) -> list:
    citations = []
    seen = set()

    for index, chunk in enumerate(chunks):
        source = chunk.get("source") or chunk.get("document") or "Unknown"
        page = chunk.get("page") or chunk.get("page_number") or "?"
        chunk_id = chunk.get("chunk_id") or chunk.get("id") or f"{source}_{page}_{index}"

        key = f"{source}_{page}_{chunk_id}"

        if key not in seen:
            seen.add(key)
            citations.append(formatCitation(source, page, chunk_id))

    return citations


def extractContexts(chunks: list) -> list:
    contexts = []

    for chunk in chunks:
        text = chunk.get("text") or chunk.get("content") or ""

        if text.strip():
            contexts.append(text.strip())

    return contexts