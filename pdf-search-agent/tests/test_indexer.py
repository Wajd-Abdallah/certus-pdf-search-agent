from app.chunker import parseAndChunk
from app.indexer import Indexer

# Use a separate, disposable directory so this script never touches
# the same ChromaDB collection the live Streamlit app uses.
TEST_PERSIST_DIR = "./data/test_chroma_db"

chunks = parseAndChunk("02a_Pflichtenheft.pdf")

indexer = Indexer(persist_directory=TEST_PERSIST_DIR)
indexer.clear()
indexer.index_chunks(chunks)

results = indexer.search("What is the goal of the project?", top_k=3)

print("Number of results:", len(results))
print(results[0])