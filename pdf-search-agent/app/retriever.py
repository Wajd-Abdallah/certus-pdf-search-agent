"""
Retrieves the most relevant chunks for a query from the Indexer.
Depends on app/indexer.py's Indexer.search() output shape:
    {"chunk_id", "text_content", "metadata", "distance"}
"""

from __future__ import annotations
import logging

from app.indexer import Indexer

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        max_distance: float | None = None,
    ) -> list[dict]:
        """
        Returns the top_k most relevant chunks for the query.

        If max_distance is set, chunks with a distance greater than this
        value are dropped (they're considered too weak to be trustworthy
        evidence). This is what allows the pipeline to abstain later when
        nothing relevant enough was found.
        """
        if not query.strip():
            logger.warning("Empty query received.")
            return []

        results = self.indexer.search(
            query=query,
            top_k=top_k
        )

        if max_distance is not None:
            before = len(results)
            results = [r for r in results if r["distance"] <= max_distance]
            dropped = before - len(results)
            if dropped:
                logger.info("Dropped %d chunk(s) above max_distance=%.3f", dropped, max_distance)

        logger.info(
            "Retrieved %d chunk(s) for query: '%s'",
            len(results),
            query,
        )

        for i, result in enumerate(results, start=1):
            preview = result["text_content"][:80].replace("\n", " ")
            logger.info(
                "  [%d] page=%s doc=%s dist=%.3f | %s...",
                i,
                result["metadata"].get("page_number"),
                result["metadata"].get("document_name"),
                result["distance"],
                preview,
            )

        return results