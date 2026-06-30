import ollama


def generateAnswer(question: str, chunks: list) -> dict:
    if not chunks:
        return {
            "answer": "I could not find enough evidence in the uploaded documents to answer this question.",
            "citations": [],
            "abstained": True,
        }

    context = buildContext(chunks)

    prompt = f"""You are a reliable PDF Search Agent. Your task is to answer questions based only on the provided document context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer using only the information from the context above.
- If the context does not contain enough information, respond with exactly:
  "I could not find enough evidence in the uploaded documents to answer this question."
- Do not make up any information.
- Be concise and clear.

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response["message"]["content"].strip()
    abstained = "could not find enough evidence" in answer.lower()

    if abstained:
        citations = []
    else:
        citations = extractCitations(chunks)

    return {
        "answer": answer,
        "citations": citations,
        "abstained": abstained,
    }


def buildContext(chunks: list) -> str:
    contextParts = []

    for chunk in chunks:
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        contextParts.append(f"[Source: {source}, Page: {page}]\n{text}")

    return "\n\n".join(contextParts)


def extractCitations(chunks: list) -> list:
    citations = []
    seen = set()

    for chunk in chunks:
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "?")
        key = f"{source}_p{page}"

        if key not in seen:
            seen.add(key)
            citations.append({"source": source, "page": page})

    return citations