"""
Indexes exactly the PDFs listed in benchmark_manifest.json into the isolated
evaluation ChromaDB collection.

Important:
- evaluation._env must be imported before app.pipeline so the evaluation
  database is used instead of the live Streamlit database.
- The collection is cleared before indexing.
- Only manifest-listed PDFs are indexed.
- Missing or invalid PDFs cause the script to stop.
- A summary file is written after successful indexing.
"""

from __future__ import annotations

import evaluation._env  # must be imported before app.pipeline

import json
from pathlib import Path
from typing import Any

from app.pipeline import indexer, process_pdf


PROJECT_ROOT = Path(__file__).resolve().parent

MANIFEST_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "data"
    / "benchmark_manifest.json"
)

INDEXING_SUMMARY_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "benchmark_indexing_summary.json"
)


def load_manifest(path: Path) -> dict[str, Any]:
    """
    Loads and validates the benchmark manifest.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark manifest not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Benchmark manifest is empty: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise ValueError(
            "Benchmark manifest must contain a JSON object."
        )

    indexed_document_ids = manifest.get(
        "indexed_document_ids"
    )

    if not isinstance(indexed_document_ids, list):
        raise ValueError(
            "Manifest field 'indexed_document_ids' "
            "must be a list."
        )

    if not indexed_document_ids:
        raise ValueError(
            "Manifest contains no indexed document IDs."
        )

    if len(indexed_document_ids) != len(
        set(indexed_document_ids)
    ):
        raise ValueError(
            "Manifest contains duplicate indexed document IDs."
        )

    pdf_directory = manifest.get("pdf_directory")

    if not isinstance(pdf_directory, str) or not pdf_directory.strip():
        raise ValueError(
            "Manifest field 'pdf_directory' "
            "must be a non-empty string."
        )

    return manifest


def resolve_pdf_directory(
    manifest: dict[str, Any],
) -> Path:
    """
    Resolves the PDF directory stored in the manifest.
    """
    configured_path = Path(
        manifest["pdf_directory"]
    )

    if configured_path.is_absolute():
        pdf_directory = configured_path
    else:
        pdf_directory = (
            PROJECT_ROOT / configured_path
        )

    if not pdf_directory.exists():
        raise FileNotFoundError(
            f"Benchmark PDF directory not found: "
            f"{pdf_directory}"
        )

    if not pdf_directory.is_dir():
        raise NotADirectoryError(
            f"Benchmark PDF path is not a directory: "
            f"{pdf_directory}"
        )

    return pdf_directory


def is_valid_pdf(path: Path) -> bool:
    """
    Performs lightweight local PDF validation.
    """
    if not path.exists() or not path.is_file():
        return False

    if path.stat().st_size < 10:
        return False

    try:
        with path.open("rb") as file:
            return file.read(4) == b"%PDF"
    except OSError:
        return False


def build_required_pdf_paths(
    manifest: dict[str, Any],
    pdf_directory: Path,
) -> list[Path]:
    """
    Builds and validates the exact list of PDFs required by the manifest.
    """
    required_paths = [
        pdf_directory / f"{document_id}.pdf"
        for document_id in manifest[
            "indexed_document_ids"
        ]
    ]

    missing_files = [
        path.name
        for path in required_paths
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required benchmark PDFs are missing: "
            + ", ".join(missing_files)
        )

    invalid_files = [
        path.name
        for path in required_paths
        if not is_valid_pdf(path)
    ]

    if invalid_files:
        raise ValueError(
            "Invalid benchmark PDFs found: "
            + ", ".join(invalid_files)
        )

    return required_paths


def write_summary(
    summary: dict[str, Any],
) -> None:
    """
    Writes the indexing summary atomically.
    """
    INDEXING_SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        INDEXING_SUMMARY_FILE.with_suffix(
            ".json.tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(
        INDEXING_SUMMARY_FILE
    )


def main() -> None:
    manifest = load_manifest(
        MANIFEST_FILE
    )

    pdf_directory = resolve_pdf_directory(
        manifest
    )

    required_pdf_paths = build_required_pdf_paths(
        manifest,
        pdf_directory,
    )

    print("=" * 72)
    print("CERTUS benchmark indexing")
    print("=" * 72)
    print(
        f"Manifest:          {MANIFEST_FILE}"
    )
    print(
        f"PDF directory:     {pdf_directory}"
    )
    print(
        f"Required PDFs:     {len(required_pdf_paths)}"
    )
    print(
        f"Collection name:   {indexer.collection_name}"
    )
    print(
        f"Chroma directory:  {indexer.persist_directory}"
    )

    print("\nClearing evaluation collection...")
    indexer.clear()

    initial_count = indexer.collection.count()

    if initial_count != 0:
        raise RuntimeError(
            "The evaluation collection was not empty "
            f"after clear(). Remaining chunks: {initial_count}"
        )

    indexed_files = []
    failed_files = []
    total_chunks_reported = 0

    for position, pdf_path in enumerate(
        required_pdf_paths,
        start=1,
    ):
        print(
            f"\n[{position}/{len(required_pdf_paths)}] "
            f"Indexing {pdf_path.name} ..."
        )

        result = process_pdf(
            str(pdf_path)
        )

        success = bool(
            result.get("success")
        )

        num_chunks = int(
            result.get("num_chunks", 0)
            or 0
        )

        message = str(
            result.get("message", "")
        )

        print(
            f"  success={success}, "
            f"chunks={num_chunks}"
        )
        print(
            f"  message={message}"
        )

        if not success:
            failed_files.append(
                {
                    "file": pdf_path.name,
                    "message": message,
                }
            )

            break

        if num_chunks <= 0:
            failed_files.append(
                {
                    "file": pdf_path.name,
                    "message": (
                        "Indexing reported success but "
                        "created no chunks."
                    ),
                }
            )

            break

        indexed_files.append(
            {
                "file": pdf_path.name,
                "num_chunks": num_chunks,
            }
        )

        total_chunks_reported += num_chunks

    if failed_files:
        summary = {
            "success": False,
            "manifest_file": str(
                MANIFEST_FILE
            ),
            "pdf_directory": str(
                pdf_directory
            ),
            "expected_pdf_count": len(
                required_pdf_paths
            ),
            "indexed_pdf_count": len(
                indexed_files
            ),
            "indexed_files": indexed_files,
            "failed_files": failed_files,
            "reported_total_chunks": (
                total_chunks_reported
            ),
            "collection_count": (
                indexer.collection.count()
            ),
            "collection_name": (
                indexer.collection_name
            ),
            "persist_directory": (
                indexer.persist_directory
            ),
        }

        write_summary(summary)

        raise RuntimeError(
            "Benchmark indexing stopped because "
            f"{failed_files[0]['file']} failed: "
            f"{failed_files[0]['message']}"
        )

    final_collection_count = (
        indexer.collection.count()
    )

    if len(indexed_files) != len(
        required_pdf_paths
    ):
        raise RuntimeError(
            "Not all required PDFs were indexed. "
            f"Expected: {len(required_pdf_paths)}; "
            f"indexed: {len(indexed_files)}."
        )

    if final_collection_count <= 0:
        raise RuntimeError(
            "Indexing completed without errors, but "
            "the evaluation collection is empty."
        )

    if (
        final_collection_count
        != total_chunks_reported
    ):
        raise RuntimeError(
            "The ChromaDB collection count does not "
            "match the total number of chunks reported "
            "by process_pdf(). "
            f"Reported: {total_chunks_reported}; "
            f"stored: {final_collection_count}."
        )

    average_chunks_per_pdf = (
        final_collection_count
        / len(indexed_files)
    )

    summary = {
        "success": True,
        "benchmark_name": manifest.get(
            "benchmark_name"
        ),
        "benchmark_version": manifest.get(
            "benchmark_version"
        ),
        "manifest_file": str(
            MANIFEST_FILE
        ),
        "pdf_directory": str(
            pdf_directory
        ),
        "expected_pdf_count": len(
            required_pdf_paths
        ),
        "indexed_pdf_count": len(
            indexed_files
        ),
        "indexed_files": indexed_files,
        "reported_total_chunks": (
            total_chunks_reported
        ),
        "collection_count": (
            final_collection_count
        ),
        "average_chunks_per_pdf": round(
            average_chunks_per_pdf,
            2,
        ),
        "collection_name": (
            indexer.collection_name
        ),
        "persist_directory": (
            indexer.persist_directory
        ),
    }

    write_summary(summary)

    print("\n" + "=" * 72)
    print("Benchmark indexing completed successfully")
    print("=" * 72)
    print(
        f"Indexed PDFs:          "
        f"{len(indexed_files)}"
    )
    print(
        f"Stored chunks:         "
        f"{final_collection_count}"
    )
    print(
        f"Average chunks/PDF:    "
        f"{average_chunks_per_pdf:.2f}"
    )
    print(
        f"Indexing summary:      "
        f"{INDEXING_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()