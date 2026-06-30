from __future__ import annotations
import logging
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from app.chunker import TextChunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Indexer:
    def __init__(
        self,
        collection_name: str = "pdf_chunks",
        persist_directory: str = "./data/chroma_db",
    ) -> None:
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(path=persist_directory)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

        logger.info(
            "Indexer ready — collection '%s' at '%s'",
            collection_name,
            persist_directory,
        )

    def index_chunks(self, chunks: List[TextChunk]) -> None:
        if not chunks:
            logger.warning("No chunks to index.")
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text_content for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info("Indexed %s chunks.", len(chunks))

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=max(top_k * 8, 20),
        )

        raw_output = []

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(documents)):
            metadata = metadatas[i] or {}

            raw_output.append(
                {
                    "text": documents[i],
                    "document_id": metadata.get("document_id"),
                    "page_number": metadata.get("page_number"),
                    "distance": distances[i],
                }
            )

        seen_texts = set()
        deduplicated = []

        for item in raw_output:
            text_key = item["text"].strip()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                deduplicated.append(item)

        query_lower = query.lower()
        prioritize_first_page = any(
            keyword in query_lower for keyword in ["title", "author", "abstract"]
        )

        if prioritize_first_page:
            deduplicated.sort(
                key=lambda x: (
                    x["page_number"] if x["page_number"] is not None else 9999,
                    x["distance"],
                )
            )
        else:
            deduplicated.sort(key=lambda x: x["distance"])

        return deduplicated[:top_k]