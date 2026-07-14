from app.citation_formatter import format_citations

chunks = [
    {"chunk_id": "abc123", "text_content": "...", "metadata": {"document_name": "01_Angebot.pdf", "page_number": 15}},
    {"chunk_id": "def456", "text_content": "...", "metadata": {"document_name": "01_Angebot.pdf", "page_number": 12}},
    {"chunk_id": "ghi789", "text_content": "...", "metadata": {"document_name": "01_Angebot.pdf", "page_number": 22}},
]

answer = "There are four students working on this project. [01_Angebot.pdf:15]"

result = format_citations(answer, chunks)
print(result)
