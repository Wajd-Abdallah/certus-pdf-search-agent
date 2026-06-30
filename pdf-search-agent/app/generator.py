import ollama

from app.citation_formatter import formatCitations, extractContexts
from app.abstention import ABSTAIN_MESSAGE, isAbstained, buildAbstentionOutput
from app.schemas import Prediction


def generateAnswer(question: str, chunks: list) -> dict:
    if not chunks:
        return buildAbstentionOutput(question)

    context = buildContext(chunks)

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
        )
        answer = response["message"]["content"].strip()

    except Exception:
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


def buildContext(chunks: list) -> str:
    contextParts = []

    for index, chunk in enumerate(chunks):
        source = chunk.get("source") or chunk.get("document") or "Unknown"
        page = chunk.get("page") or chunk.get("page_number") or "?"
        chunk_id = chunk.get("chunk_id") or chunk.get("id") or f"chunk_{index}"
        text = chunk.get("text") or chunk.get("content") or ""

        contextParts.append(
            f"[Document: {source}, Page: {page}, Chunk ID: {chunk_id}]\n{text}"
        )

    return "\n\n".join(contextParts)