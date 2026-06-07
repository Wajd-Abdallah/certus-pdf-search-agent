# from __future__ import annotations

import logging
from typing import List

import chromadb
from chromadb.utils import embedding_functions

from chunker import TextChunk

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
            n_results=top_k,
        )

        output = []

        for i in range(len(results["documents"][0])):
            output.append(
                {
                    "text": results["documents"][0][i],
                    "document_id": results["metadatas"][0][i]["document_id"],
                    "page_number": results["metadatas"][0][i]["page_number"],
                    "score": results["distances"][0][i],
                }
            )

        return output

    def clear(self) -> None:

        self.collection.delete(
            where={"chunk_index": {"$gte": 0}}
        )

        logger.info("Index cleared.")Chunks speichern / indexiereng