 # PDF einlesen & Text extrahieren
 # app/parser.py
import fitz  # PyMuPDF

def parse_pdf(file_path: str) -> list[dict]:
    """
    Liest eine PDF-Datei und gibt eine Liste von Seiten zurück.
    Jede Seite ist ein dict mit 'page_number' und 'text'.
    """
    doc = fitz.open(file_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_number": i + 1,
            "text": text.strip()
        })

    return pages