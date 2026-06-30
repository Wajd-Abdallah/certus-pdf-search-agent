from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import uuid
import fitz


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
                {
                    page_number: page_text
                }
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

        if path.stat().st_size == 0:
            raise ValueError("The PDF file is empty.")

        # -----------------------------
        # Create metadata
        # -----------------------------
        document = DocumentData(
            document_id=str(uuid.uuid4()),
            file_name=path.name,
            upload_date=datetime.now().isoformat(),
            file_size=path.stat().st_size,
            file_type="application/pdf",
            storage_path=str(path.resolve())
        )

        pages: dict[int, str] = {}

        # -----------------------------
        # Extract text
        # -----------------------------
        try:
            with fitz.open(file_path) as pdf:

                for page_number, page in enumerate(pdf, start=1):

                    text = page.get_text("text").strip()

                    # Skip empty pages
                    if not text:
                        continue

                    pages[page_number] = text

        except Exception as error:
            raise ValueError(f"PDF parsing failed: {error}")

        if not pages:
            raise ValueError("No readable text found in the PDF.")

        return document, pages

    # ======================================================

    def parse_multiple(
        self,
        file_paths: list[str]
    ) -> list[tuple[DocumentData, dict[int, str]]]:
        """
        Parses multiple PDF files.

        Invalid PDFs are skipped with a warning.
        """

        results = []

        for file_path in file_paths:

            try:
                results.append(self.parse(file_path))

            except Exception as error:
                print(f"Warning: Skipping '{file_path}': {error}")

        return results