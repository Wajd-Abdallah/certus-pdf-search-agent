from app.generator import generateAnswer


def testNormalAnswer():
    #Test 1: Normal answer with relevant chunks
    chunks = [
        {'text': 'RAG stands for Retrieval Augmented Generation. It retrieves relevant documents before generating an answer.', 'source': 'paper1.pdf', 'page': 3},
        {'text': 'RAG reduces hallucinations by grounding answers in real documents.', 'source': 'paper1.pdf', 'page': 5}
    ]

    result = generateAnswer('What is RAG?', chunks)

    print('Test 1: Normal Answer')
    print('Answer:', result['answer'])
    print('Citations:', result['citations'])
    print('Abstained:', result['abstained'])

    assert result['abstained'] == False
    assert len(result['citations']) > 0
    print('Test 1 passed!\n')


def testAbstentionNoChunks():
    #Test 2:Abstention (no chunks ) 
    result = generateAnswer('What is the capital of France?', [])

    print('=== Test 2: Abstention - empty chunks ===')
    print('Answer:', result['answer'])
    print('Citations:', result['citations'])
    print('Abstained:', result['abstained'])

    assert result['abstained'] == True
    assert result['citations'] == []
    print('Test 2 passed!\n')


def testAbstentionIrrelevantChunks():
    # Test 3: Abstention (chunks exist but question not answerable)
    chunks = [
        {'text': 'RAG stands for Retrieval Augmented Generation.', 'source': 'paper1.pdf', 'page': 3}
    ]

    result = generateAnswer('What is the capital of France?', chunks)

    print('=== Test 3: Abstention - irrelevant chunks ===')
    print('Answer:', result['answer'])
    print('Citations:', result['citations'])
    print('Abstained:', result['abstained'])

    assert result['abstained'] == True
    assert result['citations'] == []
    print('Test 3 passed!\n')

def testConvertChunksFromDict():
    # Test convertChunks with dictionary input
    from app.pipeline import convertChunks

    fakeChunks = [
        {'text': 'RAG is a method.', 'source': 'paper1.pdf', 'page': 3}
    ]

    converted = convertChunks(fakeChunks)

    print('=== Test 4: convertChunks from dict ===')
    print('Converted:', converted)

    assert converted[0]['text'] == 'RAG is a method.'
    assert converted[0]['source'] == 'paper1.pdf'
    assert converted[0]['page'] == 3
    print('Test 4 passed!\n')


def testConvertChunksFromObject():
    # Test convertChunks with TextChunk-like object input
    from app.pipeline import convertChunks

    class FakeChunk:
        text_content = 'RAG reduces hallucinations.'
        document_id = 'paper2.pdf'
        page_number = 5

    converted = convertChunks([FakeChunk()])

    print('=== Test 5: convertChunks from object ===')
    print('Converted:', converted)

    assert converted[0]['text'] == 'RAG reduces hallucinations.'
    assert converted[0]['source'] == 'paper2.pdf'
    assert converted[0]['page'] == 5
    print('Test 5 passed!\n')


def testFullPipelineWithFakeChunks():
    # Test full pipeline: convert chunks then generate answer
    from app.pipeline import convertChunks
    from app.generator import generateAnswer

    fakeChunks = [
        {'text': 'RAG stands for Retrieval Augmented Generation.', 'source': 'paper1.pdf', 'page': 3}
    ]

    converted = convertChunks(fakeChunks)
    result = generateAnswer('What is RAG?', converted)

    print('=== Test 6: Full pipeline with fake chunks ===')
    print('Answer:', result['answer'])
    print('Citations:', result['citations'])
    print('Abstained:', result['abstained'])

    assert result['abstained'] == False
    assert len(result['citations']) > 0
    print('Test 6 passed!\n')

if __name__ == '__main__':
    testNormalAnswer()
    testAbstentionNoChunks()
    testAbstentionIrrelevantChunks()
    testConvertChunksFromDict()
    testConvertChunksFromObject()
    testFullPipelineWithFakeChunks()
    print('All tests passed!')
