import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import uuid

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


# ==========================================================
# DocumentData
# ==========================================================
@dataclass
class DocumentData:
    """
    Stores metadata of one uploaded PDF document.
    """
    document_id: str
    file_name: str
    upload_date: str
    file_size: int
    file_type: str
    storage_path: str
    num_pages: int


# ==========================================================
# PDFParser
# ==========================================================
class PDFParser:
    """
    Validates PDF files and extracts text page by page.
    """

    def parse(self, file_path: str) -> tuple[DocumentData, dict[int, str]]:
        """
        Parses one PDF.

        Returns:
            (
                DocumentData,
                {page_number: page_text, ...}
            )
        """
        path = Path(file_path)

        # -----------------------------
        # Validation
        # -----------------------------
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are allowed.")

        file_size = path.stat().st_size
        if file_size == 0:
            raise ValueError("The PDF file is empty.")

        # -----------------------------
        # Extract text
        # -----------------------------
        pages: dict[int, str] = {}

        try:
            with fitz.open(file_path) as pdf:
                if pdf.is_encrypted:
                    raise ValueError("The PDF is password-protected and cannot be read.")

                for page_number, page in enumerate(pdf, start=1):
                    text = page.get_text("text").strip()
                    if not text:
                        continue  # skip empty pages (e.g. scanned images with no OCR)
                    pages[page_number] = text

                num_pages = pdf.page_count
        except ValueError:
            raise
        except Exception as error:
            raise ValueError(f"PDF parsing failed: {error}") from error

        if not pages:
            raise ValueError("No readable text found in the PDF.")

        # -----------------------------
        # Create metadata
        # -----------------------------
        document = DocumentData(
            document_id=str(uuid.uuid4()),
            file_name=path.name,
            upload_date=datetime.now().isoformat(),
            file_size=file_size,
            file_type="application/pdf",
            storage_path=str(path.resolve()),
            num_pages=num_pages,
        )

        return document, pages

    def parse_multiple(
        self,
        file_paths: list[str]
    ) -> list[tuple[DocumentData, dict[int, str]]]:
        """
        Parses multiple PDF files.
        Invalid PDFs are skipped with a warning (not a crash).
        """
        results = []
        for file_path in file_paths:
            try:
                results.append(self.parse(file_path))
            except Exception as error:
                logger.warning("Skipping '%s': %s", file_path, error)
        return results