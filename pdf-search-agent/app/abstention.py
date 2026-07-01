ABSTAIN_MESSAGE = "I could not find enough evidence in the uploaded documents to answer this question."


def isAbstained(answer: str) -> bool:
    if not answer:
        return True

    return ABSTAIN_MESSAGE.lower() in answer.lower()


def buildAbstentionOutput(question: str, reason: str = "insufficient_context") -> dict:
    return {
        "question": question,
        "answer": ABSTAIN_MESSAGE,
        "citations": [],
        "abstained": True,
        "abstention_reason": reason,
        "retrieved_contexts": []
    }