from app.generator import generateAnswer


def testNormalAnswer():
    chunks = [
        {
            "text": "RAG stands for Retrieval Augmented Generation. It retrieves relevant documents before generating an answer.",
            "source": "paper1.pdf",
            "page": 3
        },
        {
            "text": "RAG reduces hallucinations by grounding answers in real documents.",
            "source": "paper1.pdf",
            "page": 5
        }
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


def testAbstentionNoChunks():
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


def testOutputSchema():
    chunks = [
        {
            "text": "RAG stands for Retrieval Augmented Generation.",
            "source": "paper1.pdf",
            "page": 3
        }
    ]

    result = generateAnswer("What is RAG?", chunks)

    print("Test 3: Output schema")
    print(result)

    expected_keys = {
        "question",
        "answer",
        "citations",
        "abstained",
        "abstention_reason",
        "retrieved_contexts"
    }

    assert set(result.keys()) == expected_keys
    assert isinstance(result["question"], str)
    assert isinstance(result["answer"], str)
    assert isinstance(result["citations"], list)
    assert isinstance(result["abstained"], bool)
    assert isinstance(result["retrieved_contexts"], list)

    print("Test 3 passed!\n")


if __name__ == "__main__":
    testNormalAnswer()
    testAbstentionNoChunks()
    testOutputSchema()
    print("All tests passed!")