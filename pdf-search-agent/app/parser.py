"""
parser.py
=========
PDF ingestion and text extraction using PyMuPDF.

Responsibilities
----------------
- Load PDF files from disk
- Extract clean text per page
- Handle broken/invalid PDFs gracefully (Q80)

Output
------
- DocumentData
- dict[int, str]  (page number -> extracted text)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Data Model
# -------------------------------------------------------------------

@dataclass
class DocumentData:
    """
    Metadata of an ingested PDF document.

    Attributes
    ----------
    document_id : str
        Unique identifier (UUID)

    file_name : str
        Original file name

    upload_date : str
        ISO-8601 timestamp of ingestion

    file_size : int
        File size in bytes

    file_type : str
        MIME type (always application/pdf)

    storage_path : str
        Absolute file path on disk
    """

    document_id: str
    file_name: str
    upload_date: str
    file_size: int
    file_type: str
    storage_path: str


# -------------------------------------------------------------------
# PDF Parser
# -------------------------------------------------------------------

class PDFParser:
    """
    PDF parser using PyMuPDF.

    Example
    -------
    >>> parser = PDFParser()
    >>> document, pages = parser.parse("example.pdf")
    >>> print(pages[1])
    """

    def parse(self, file_path: str) -> tuple[DocumentData, dict[int, str]]:
        """
        Parse a single PDF file.

        Parameters
        ----------
        file_path : str
            Path to the PDF file.

        Returns
        -------
        tuple[DocumentData, dict[int, str]]
            Document metadata and extracted page texts.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.

        ValueError
            If the file is invalid or unreadable.
        """

        path = Path(file_path).resolve()

        # -----------------------------------------------------------
        # Basic validation
        # -----------------------------------------------------------

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Expected a PDF file, got '{path.suffix}'"
            )

        if path.stat().st_size == 0:
            raise ValueError(f"PDF '{path.name}' is empty.")

        # -----------------------------------------------------------
        # Create document metadata
        # -----------------------------------------------------------

        document = DocumentData(
            document_id=str(uuid.uuid4()),
            file_name=path.name,
            upload_date=datetime.utcnow().isoformat() + "Z",
            file_size=path.stat().st_size,
            file_type="application/pdf",
            storage_path=str(path),
        )

        # -----------------------------------------------------------
        # Open and parse PDF
        # -----------------------------------------------------------

        try:
            with fitz.open(str(path)) as doc:

                if doc.page_count == 0:
                    raise ValueError(
                        f"PDF '{path.name}' contains no pages."
                    )

                pages: dict[int, str] = {}

                for page_num in range(doc.page_count):

                    try:
                        page = doc.load_page(page_num)

                        # Extract plain text
                        text = page.get_text("text")

                        # Clean whitespace
                        cleaned_text = " ".join(text.split())

                        # Skip empty pages
                        if cleaned_text:
                            pages[page_num + 1] = cleaned_text

                    except Exception as exc:
                        logger.warning(
                            "Could not read page %s in '%s': %s",
                            page_num + 1,
                            path.name,
                            exc,
                        )

                if not pages:
                    raise ValueError(
                        f"No readable text found in '{path.name}'."
                    )

                logger.info(
                    "Successfully parsed '%s' (%s pages)",
                    path.name,
                    len(pages),
                )

                return document, pages

        except fitz.FileDataError as exc:
            # ← NUR echte PyMuPDF-Fehler werden hier gefangen,
            #   nicht unsere eigenen ValueErrors von oben
            raise ValueError(
                f"Could not open PDF '{path.name}': {exc}"
            ) from exc

    # -------------------------------------------------------------------

    def parse_multiple(
        self,
        file_paths: list[str],
    ) -> list[tuple[DocumentData, dict[int, str]]]:
        """
        Parse multiple PDF files.

        Invalid files are skipped instead of crashing the pipeline.

        Parameters
        ----------
        file_paths : list[str]
            List of PDF paths.

        Returns
        -------
        list[tuple[DocumentData, dict[int, str]]]
            Successfully parsed documents.
        """

        results = []

        for file_path in file_paths:

            try:
                parsed = self.parse(file_path)
                results.append(parsed)

            except (FileNotFoundError, ValueError) as exc:

                logger.warning(
                    "Skipping '%s': %s",
                    file_path,
                    exc,
                )

        return results