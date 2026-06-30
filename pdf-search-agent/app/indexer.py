import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.chunker import TextChunk


class Indexer:
    def __init__(
        self,
        collection_name: str = "pdf_chunks",
        persist_directory: str = "./data/chroma_db"
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        self.embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )

    def index_chunks(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            raise ValueError("No chunks provided for indexing.")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text_content for chunk in chunks]

        metadatas = [
            {
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "token_count": chunk.token_count,
                "page_number": chunk.page_number,
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int) -> list[dict]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        output = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            output.append(
                {
                    "chunk_id": chunk_id,
                    "text_content": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return output

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )