from __future__ import annotations
import logging
from typing import List
from app.chunker import parseAndChunk
from app.generator import generateAnswer
from app.indexer import Indexer
from app.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global shared instances for the baseline
indexer = Indexer()
retriever = Retriever(indexer=indexer)


def processPdf(filePath: str) -> dict:
    try:
        chunks = parseAndChunk(filePath)
        if not chunks:
            return {
                "success": False,
                "message": f"No readable text could be extracted from {filePath}",
                "num_chunks": 0,
            }
        indexer.index_chunks(chunks)
        logger.info("Successfully processed and indexed %s chunks.", len(chunks))
        return {
            "success": True,
            "message": f"Successfully processed {len(chunks)} chunks from {filePath}",
            "num_chunks": len(chunks),
        }
    except Exception as e:
        logger.exception("Error while processing PDF.")
        return {
            "success": False,
            "message": f"Error processing PDF: {str(e)}",
            "num_chunks": 0,
        }


def convertChunks(rawChunks: List[dict]) -> list:
    """
    Normalize retrieval output into generator input format:
    {'text', 'source', 'page'}
    """
    converted = []
    for chunk in rawChunks:
        metadata = chunk.get("metadata", {})
        converted.append(
            {
                "text": chunk.get("text_content", ""),
                "source": metadata.get("document_name", "Unknown"),
                "page": metadata.get("page_number", "?"),
            }
        )
    return converted


def answerQuestion(question: str, topK: int = 5) -> dict:
    """
    Full QA pipeline:
    retrieve -> convert -> generate
    """
    try:
        rawChunks = retriever.retrieve(question, top_k=topK)
        chunks = convertChunks(rawChunks)
        result = generateAnswer(question, chunks)
        return result
    except Exception as e:
        logger.exception("Error while answering question.")
        return {
            "answer": f"An error occurred: {str(e)}",
            "citations": [],
            "abstained": True,
        }