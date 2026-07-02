"""
Generates a grounded answer from retrieved chunks using a local Ollama model.
Depends on app/retriever.py's output shape:
    {"chunk_id", "text_content", "metadata": {..., "document_name", "page_number"}, "distance"}
"""

import logging

import ollama

from app.citation_formatter import formatCitations, extractContexts
from app.abstention import ABSTAIN_MESSAGE, isAbstained, buildAbstentionOutput
from app.schemas import Prediction

logger = logging.getLogger(__name__)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    if not chunks:
        return buildAbstentionOutput(question)

    context = build_context(chunks)

    prompt = f"""You are a reliable PDF Search Agent. Your task is to answer questions based only on the provided document context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer only using the information from the context above.
- Do not use external knowledge.
- Do not make up information.
- If the context does not contain enough information, respond exactly with:
"{ABSTAIN_MESSAGE}"
- Keep the answer concise and clear.

Answer:"""

    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={"temperature": 0},  # deterministic output for reproducibility
        )
        answer = response["message"]["content"].strip()
    except Exception:
        logger.exception("Generation failed for question: '%s'", question)
        return buildAbstentionOutput(question, reason="generation_error")

    abstained = isAbstained(answer)
    retrieved_contexts = extractContexts(chunks)
    citations = formatCitations(chunks) if not abstained else []

    prediction = Prediction(
        question=question,
        answer=answer,
        citations=citations,
        abstained=abstained,
        abstention_reason="insufficient_context" if abstained else None,
        retrieved_contexts=retrieved_contexts,
    )

    return prediction.toDict()


def build_context(chunks: list[dict]) -> str:
    context_parts = []
    for index, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        source = metadata.get("document_name") or "Unknown"
        page = metadata.get("page_number", "?")
        chunk_id = chunk.get("chunk_id") or f"chunk_{index}"
        text = chunk.get("text_content") or ""

        context_parts.append(
            f"[Document: {source}, Page: {page}, Chunk ID: {chunk_id}]\n{text}"
        )
    return "\n\n".join(context_parts)


# Backward-compatible aliases in case other files still import the old camelCase names.
generateAnswer = generate_answer
buildContext = build_context