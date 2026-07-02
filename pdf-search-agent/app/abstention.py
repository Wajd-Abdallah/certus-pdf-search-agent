"""
Defines the abstention message and logic for detecting when the model
declined to answer due to insufficient evidence.
"""

ABSTAIN_MESSAGE = "I could not find enough evidence in the uploaded documents to answer this question."

# Short, distinctive anchor phrase used for matching. Kept separate from the
# full ABSTAIN_MESSAGE because local LLMs (e.g. Llama 3.2) don't always
# reproduce instructed sentences verbatim -- they may paraphrase slightly.
# Matching on a shorter, distinctive fragment is more robust than matching
# the whole sentence, while still being unlikely to false-positive.
_ABSTAIN_ANCHOR = "could not find enough evidence"


def is_abstained(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    return _ABSTAIN_ANCHOR in answer.lower()


def build_abstention_output(question: str, reason: str = "insufficient_context") -> dict:
    return {
        "question": question,
        "answer": ABSTAIN_MESSAGE,
        "citations": [],
        "abstained": True,
        "abstention_reason": reason,
        "retrieved_contexts": []
    }


# Backward-compatible aliases in case other files still import the old camelCase names.
isAbstained = is_abstained
buildAbstentionOutput = build_abstention_output