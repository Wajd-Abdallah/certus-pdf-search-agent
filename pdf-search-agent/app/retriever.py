from __future__ import annotations

import logging
from typing import List

from indexer import Indexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:

    def __init__(self, indexer: Indexer) -> None:
        self.indexer = indexer

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:

        if not query.strip():
            logger.warning("Empty query received.")
            return []

        results = self.indexer.search(query=query, top_k=top_k)

        logger.info(
            "Retrieved %s chunks for query: '%s'",
            len(results),
            query,
        )

        return results# Relevante Chunks finden