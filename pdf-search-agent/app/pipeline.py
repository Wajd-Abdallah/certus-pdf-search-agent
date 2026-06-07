import logging
from app.generator import generateAnswer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def processPdf(filePath: str) -> dict:
    # Process a PDF file through the full pipeline
    try:
        from app.chunker import parseAndChunk
        chunks = parseAndChunk(filePath)

        if not chunks:
            return {
                'success': False,
                'message': f'No text could be extracted from {filePath}'
            }

        # Step 2: Index chunks (connect when indexer.py is ready)
        try:
            from app.indexer import index_chunks
            index_chunks(chunks)
            logger.info("Chunks indexed successfully.")
        except Exception as e:
            logger.warning(f"Indexer not ready yet: {e}")

        return {
            'success': True,
            'message': f'Successfully processed {len(chunks)} chunks from {filePath}'
        }

    except Exception as e:
        return {
            'success': False,
            'message': f'Error processing PDF: {str(e)}'
        }


def convertChunks(rawChunks: list) -> list:
    # Convert chunks to generator format: {'text', 'source', 'page'}
    # Handles both TextChunk objects and plain dictionaries.
    converted = []

    for chunk in rawChunks:
        # If chunk is a TextChunk object
        if hasattr(chunk, 'text_content'):
            converted.append({
                'text': chunk.text_content,
                'source': chunk.document_id,
                'page': chunk.page_number
            })
        # If chunk is already a dictionary
        elif isinstance(chunk, dict):
            converted.append({
                'text': chunk.get('text', chunk.get('text_content', '')),
                'source': chunk.get('source', chunk.get('document_id', 'Unknown')),
                'page': chunk.get('page', chunk.get('page_number', '?'))
            })

    return converted


def answerQuestion(question: str, topK: int = 5) -> dict:
    # Answer a question based on indexed documents.
    try:
        # Step 1: Retrieve relevant chunks
        try:
            from app.retriever import retrieve_chunks
            rawChunks = retrieve_chunks(question, topK)
        except Exception as e:
            logger.warning(f"Retriever not ready yet: {e}")
            rawChunks = []

        # Step 2: Convert chunks to generator format
        chunks = convertChunks(rawChunks)

        # Step 3: Generate answer
        result = generateAnswer(question, chunks)

        return result

    except Exception as e:
        return {
            'answer': f'An error occurred: {str(e)}',
            'citations': [],
            'abstained': True
        }