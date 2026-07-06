import evaluation._env
from pathlib import Path
from app.pipeline import process_pdf

pdf_dir = Path("evaluation/data/open_rag_bench/pdfs")

for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    print(f"Indexing {pdf_path.name} ...")
    result = process_pdf(str(pdf_path))
    print(f"  success={result['success']}, chunks={result['num_chunks']}, message={result['message']}")
