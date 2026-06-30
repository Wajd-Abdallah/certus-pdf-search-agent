from app.chunker import parseAndChunk

chunks = parseAndChunk("02a_Pflichtenheft.pdf")

print("Number of chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0])