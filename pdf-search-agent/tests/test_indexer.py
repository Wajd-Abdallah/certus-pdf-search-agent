from app.chunker import parseAndChunk
from app.indexer import Indexer

chunks = parseAndChunk("02a_Pflichtenheft.pdf")

indexer = Indexer()
indexer.clear()
indexer.index_chunks(chunks)

results = indexer.search("What is the goal of the project?", top_k=3)

print("Number of results:", len(results))
print(results[0])