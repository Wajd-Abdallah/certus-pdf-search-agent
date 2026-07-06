from app.generator import generateAnswer


def test_normal_answer():
    chunks = [
        {
            "text_content": "RAG stands for Retrieval Augmented Generation. It retrieves relevant documents before generating an answer.",
            "metadata": {"document_name": "paper1.pdf", "page_number": 3},
        },
        {
            "text_content": "RAG reduces hallucinations by grounding answers in real documents.",
            "metadata": {"document_name": "paper1.pdf", "page_number": 5},
        },
    ]

    result = generateAnswer("What is RAG?", chunks)

    print("Test 1: Normal Answer")
    print("Answer:", result["answer"])
    print("Citations:", result["citations"])
    print("Abstained:", result["abstained"])

    assert result["question"] == "What is RAG?"
    assert isinstance(result["answer"], str)
    assert isinstance(result["citations"], list)
    assert isinstance(result["abstained"], bool)
    assert "abstention_reason" in result
    assert "retrieved_contexts" in result
    assert len(result["retrieved_contexts"]) > 0
    print("Test 1 passed!\n")


def test_abstention_no_chunks():
    result = generateAnswer("What is the capital of France?", [])

    print("Test 2: Abstention - empty chunks")
    print("Answer:", result["answer"])
    print("Citations:", result["citations"])
    print("Abstained:", result["abstained"])

    assert result["abstained"] is True
    assert result["citations"] == []
    assert result["abstention_reason"] == "insufficient_context"
    assert result["retrieved_contexts"] == []
    print("Test 2 passed!\n")


def test_output_schema():
    chunks = [
        {
            "text_content": "RAG stands for Retrieval Augmented Generation.",
            "metadata": {"document_name": "paper1.pdf", "page_number": 3},
        }
    ]

    result = generateAnswer("What is RAG?", chunks)

    print("Test 3: Output schema")
    print(result)

    # Subset check, not strict equality: generate_answer legitimately gained
    # prompt_tokens/completion_tokens fields for efficiency tracking. New
    # fields being added over time shouldn't break this test -- we only
    # care that the core required keys are always present.
    required_keys = {
        "question",
        "answer",
        "citations",
        "abstained",
        "abstention_reason",
        "retrieved_contexts",
    }
    assert required_keys.issubset(result.keys())

    assert isinstance(result["question"], str)
    assert isinstance(result["answer"], str)
    assert isinstance(result["citations"], list)
    assert isinstance(result["abstained"], bool)
    assert isinstance(result["retrieved_contexts"], list)
    print("Test 3 passed!\n")


if __name__ == "__main__":
    test_normal_answer()
    test_abstention_no_chunks()
    test_output_schema()
    print("All tests passed!")