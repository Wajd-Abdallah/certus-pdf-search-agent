"""
Top-level pipeline: ties parser -> chunker -> indexer -> retriever -> generator
together. This is the main entry point used by the UI and evaluation scripts.

Configuration (chunk size, top_k, model names, persist directory, etc.) is
loaded from configs/baseline.yaml via app/config.py. If the config file is
missing or incomplete, built-in defaults are used instead (with a warning),
so the app still runs.
"""

from __future__ import annotations
import logging

from app.chunker import parse_and_chunk, RecursiveChunker
from app.config import load_config
from app.generator import generate_answer
from app.indexer import Indexer
from app.retriever import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()
logger.info("Loaded config: run='%s'", config["run"]["name"])

# Global shared instances for the baseline (single-user, single-session assumption).
# NOTE: revisit this if we later need per-session isolation (e.g. multiple users,
# or tests that expect a clean index each run).
indexer = Indexer(
    collection_name=config["index"]["collection_name"],
    persist_directory=config["index"]["persist_directory"],
)
retriever = Retriever(indexer=indexer)

# Default chunker built from config. Passed explicitly to parse_and_chunk so
# chunk_size/overlap actually come from configs/baseline.yaml instead of the
# chunker's own hardcoded defaults.
_default_chunker = RecursiveChunker(
    chunk_size=config["chunking"]["chunk_size"],
    overlap=config["chunking"]["chunk_overlap"],
)


def process_pdf(file_path: str) -> dict:
    try:
        chunks = parse_and_chunk(file_path, chunker=_default_chunker)

        # NOTE: parse_and_chunk() currently raises ValueError instead of
        # returning an empty list when no chunks could be created, so this
        # branch is effectively unreachable today. Kept as a defensive
        # guard in case that behavior changes.
        if not chunks:
            return {
                "success": False,
                "message": f"No readable text could be extracted from {file_path}",
                "num_chunks": 0,
            }

        indexer.index_chunks(chunks)
        logger.info("Successfully processed and indexed %s chunks.", len(chunks))

        return {
            "success": True,
            "message": f"Successfully processed {len(chunks)} chunks from {file_path}",
            "num_chunks": len(chunks),
        }

    except Exception as e:
        logger.exception("Error while processing PDF.")
        return {
            "success": False,
            "message": f"Error processing PDF: {str(e)}",
            "num_chunks": 0,
        }


def answer_question(question: str, top_k: int | None = None) -> dict:
    """
    Full QA pipeline: retrieve -> generate.

    Note: retriever.retrieve() already returns chunks in the shape that
    generate_answer() expects ({"text_content", "metadata": {...}, "distance"}),
    so no conversion step is needed here.
    """
    if top_k is None:
        top_k = config["retrieval"]["top_k"]

    try:
        raw_chunks = retriever.retrieve(question, top_k=top_k)
        result = generate_answer(question, raw_chunks)
        return result

    except Exception as e:
        logger.exception("Error while answering question.")
        return {
            "question": question,
            "answer": f"An error occurred: {str(e)}",
            "citations": [],
            "abstained": True,
            "abstention_reason": "generation_error",
            "retrieved_contexts": [],
        }


# Backward-compatible aliases in case other files still import the old camelCase names.
processPdf = process_pdf
answerQuestion = answer_question