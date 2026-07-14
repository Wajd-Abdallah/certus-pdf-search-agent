"""
Builds reproducible Open RAGBench evaluation datasets for CERTUS.

The script creates two benchmark files:

1. Retrieval benchmark:
   - 400 questions total
   - 300 answerable questions
   - 100 abstention questions

2. Full-pipeline benchmark:
   - 120 questions total
   - 90 answerable questions
   - 30 abstention questions
   - selected as a deterministic subset of the retrieval benchmark

The script also creates a manifest containing the exact PDFs that must be
indexed. This prevents evaluation results from depending on unrelated PDF
files that happen to exist in the benchmark directory.

Generated files:
- evaluation/data/benchmark_retrieval_400.json
- evaluation/data/benchmark_full_120.json
- evaluation/data/benchmark_subset.json
- evaluation/data/benchmark_manifest.json

The compatibility file benchmark_subset.json contains the same records as
benchmark_full_120.json so existing evaluation code continues to work until
the benchmark loader is updated.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


# ==========================================================
# Configuration
# ==========================================================

RANDOM_SEED = 42

RETRIEVAL_ANSWERABLE_COUNT = 300
RETRIEVAL_ABSTENTION_COUNT = 100

FULL_ANSWERABLE_COUNT = 90
FULL_ABSTENTION_COUNT = 30

# Limit the number of selected questions per positive document.
# This increases topic and document diversity.
MAX_QUESTIONS_PER_POSITIVE_DOCUMENT = 6

# Additional indexed PDFs that have no positive benchmark questions.
# They make retrieval more realistic by adding irrelevant document noise.
NUM_DISTRACTOR_DOCUMENTS = 10

DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT_SECONDS = 90

PROJECT_ROOT = Path(__file__).resolve().parent

METADATA_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "data"
    / "open_rag_bench"
    / "pdf"
    / "arxiv"
)

PDF_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "data"
    / "open_rag_bench"
    / "pdfs"
)

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "data"

RETRIEVAL_BENCHMARK_FILE = (
    OUTPUT_DIR / "benchmark_retrieval_400.json"
)

FULL_BENCHMARK_FILE = (
    OUTPUT_DIR / "benchmark_full_120.json"
)

# Temporary compatibility file for existing evaluation scripts.
COMPATIBILITY_BENCHMARK_FILE = (
    OUTPUT_DIR / "benchmark_subset.json"
)

MANIFEST_FILE = OUTPUT_DIR / "benchmark_manifest.json"


# ==========================================================
# JSON helpers
# ==========================================================

def load_json_object(path: Path) -> dict[str, Any]:
    """
    Loads and validates a JSON object.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required metadata file does not exist: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Required metadata file is empty: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, "
            f"but found {type(data).__name__}."
        )

    return data


def write_json(path: Path, data: Any) -> None:
    """
    Writes JSON using stable and readable formatting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(path)


# ==========================================================
# Metadata extraction and validation
# ==========================================================

def extract_document_id(
    qrel_record: Any,
    query_id: str,
) -> str:
    """
    Extracts a document ID from one qrels entry.
    """
    if not isinstance(qrel_record, dict):
        raise ValueError(
            f"Invalid qrels record for query {query_id}: "
            "expected a JSON object."
        )

    document_id = qrel_record.get("doc_id")

    if document_id is None:
        raise ValueError(
            f"Missing doc_id in qrels record for query {query_id}."
        )

    document_id = str(document_id).strip()

    if not document_id:
        raise ValueError(
            f"Empty doc_id in qrels record for query {query_id}."
        )

    return document_id


def extract_question_text(
    queries: dict[str, Any],
    query_id: str,
) -> str:
    """
    Extracts a question string from queries.json.
    """
    if query_id not in queries:
        raise KeyError(
            f"Query ID {query_id} is missing from queries.json."
        )

    record = queries[query_id]

    if isinstance(record, dict):
        question = (
            record.get("query")
            or record.get("question")
            or record.get("text")
            or ""
        )
    else:
        question = record

    question = str(question).strip()

    if not question:
        raise ValueError(
            f"Question text is empty for query ID {query_id}."
        )

    return question


def extract_expected_answer(
    answers: dict[str, Any],
    query_id: str,
) -> str:
    """
    Extracts a reference answer from answers.json.
    """
    if query_id not in answers:
        raise KeyError(
            f"Query ID {query_id} is missing from answers.json."
        )

    record = answers[query_id]

    if isinstance(record, dict):
        answer = (
            record.get("answer")
            or record.get("response")
            or record.get("text")
            or record.get("ground_truth")
            or ""
        )
    elif isinstance(record, list):
        answer = " ".join(
            str(item).strip()
            for item in record
            if str(item).strip()
        )
    else:
        answer = record

    answer = str(answer).strip()

    if not answer:
        raise ValueError(
            f"Expected answer is empty for query ID {query_id}."
        )

    return answer


def extract_pdf_url(
    pdf_urls: dict[str, Any],
    document_id: str,
) -> str:
    """
    Extracts the PDF URL for one document.
    """
    if document_id not in pdf_urls:
        raise KeyError(
            f"Document ID {document_id} is missing from pdf_urls.json."
        )

    record = pdf_urls[document_id]

    if isinstance(record, dict):
        url = (
            record.get("url")
            or record.get("pdf_url")
            or record.get("link")
            or ""
        )
    else:
        url = record

    url = str(url).strip()

    if not url:
        raise ValueError(
            f"PDF URL is empty for document {document_id}."
        )

    return url


def validate_metadata(
    qrels: dict[str, Any],
    queries: dict[str, Any],
    answers: dict[str, Any],
    pdf_urls: dict[str, Any],
) -> None:
    """
    Verifies that every qrels entry references valid metadata.
    """
    missing_queries: list[str] = []
    missing_answers: list[str] = []
    missing_pdf_urls: list[str] = []
    invalid_qrels: list[str] = []

    for query_id, qrel_record in qrels.items():
        try:
            document_id = extract_document_id(
                qrel_record,
                query_id,
            )
        except ValueError:
            invalid_qrels.append(query_id)
            continue

        if query_id not in queries:
            missing_queries.append(query_id)

        if query_id not in answers:
            missing_answers.append(query_id)

        if document_id not in pdf_urls:
            missing_pdf_urls.append(document_id)

    problems = []

    if invalid_qrels:
        problems.append(
            f"{len(invalid_qrels)} invalid qrels record(s)"
        )

    if missing_queries:
        problems.append(
            f"{len(missing_queries)} query ID(s) missing from queries.json"
        )

    if missing_answers:
        problems.append(
            f"{len(missing_answers)} query ID(s) missing from answers.json"
        )

    if missing_pdf_urls:
        problems.append(
            f"{len(set(missing_pdf_urls))} document ID(s) "
            "missing from pdf_urls.json"
        )

    if problems:
        raise ValueError(
            "Open RAGBench metadata validation failed: "
            + "; ".join(problems)
        )


# ==========================================================
# PDF validation and download
# ==========================================================

def is_valid_pdf(path: Path) -> bool:
    """
    Performs lightweight PDF validation using file size and header.
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


def download_pdf(
    document_id: str,
    url: str,
    output_path: Path,
) -> None:
    """
    Downloads one PDF with retry and resumable behavior.

    A valid existing PDF is preserved and skipped.
    Incomplete downloads use a temporary .part file.
    """
    if is_valid_pdf(output_path):
        print(f"Already downloaded: {output_path.name}")
        return

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.exists():
        output_path.unlink()

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    temporary_path.unlink(missing_ok=True)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(TU Braunschweig SEP CERTUS evaluation)"
        )
    }

    last_error: Exception | None = None

    for attempt in range(
        1,
        DOWNLOAD_RETRIES + 1,
    ):
        try:
            print(
                f"Downloading {document_id} "
                f"(attempt {attempt}/{DOWNLOAD_RETRIES}) ..."
            )

            request = urllib.request.Request(
                url,
                headers=headers,
            )

            with urllib.request.urlopen(
                request,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            ) as response:
                temporary_path.write_bytes(
                    response.read()
                )

            if not is_valid_pdf(temporary_path):
                raise ValueError(
                    f"Downloaded content for {document_id} "
                    "is not a valid PDF."
                )

            temporary_path.replace(output_path)

            print(
                f"  Saved {output_path.name} "
                f"({output_path.stat().st_size / 1024:.1f} KB)"
            )

            return

        except (
            OSError,
            ValueError,
            urllib.error.URLError,
            urllib.error.HTTPError,
        ) as error:
            last_error = error
            temporary_path.unlink(missing_ok=True)

            if attempt < DOWNLOAD_RETRIES:
                wait_seconds = attempt * 2

                print(
                    f"  Download failed: {error}. "
                    f"Retrying in {wait_seconds} second(s)..."
                )

                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Could not download PDF {document_id} "
        f"after {DOWNLOAD_RETRIES} attempts: {last_error}"
    )


# ==========================================================
# Selection helpers
# ==========================================================

def group_queries_by_document(
    qrels: dict[str, Any],
) -> dict[str, list[str]]:
    """
    Groups query IDs according to their ground-truth document.
    """
    grouped: dict[str, list[str]] = defaultdict(list)

    for query_id, qrel_record in qrels.items():
        document_id = extract_document_id(
            qrel_record,
            query_id,
        )

        grouped[document_id].append(query_id)

    for document_id in grouped:
        grouped[document_id] = sorted(
            grouped[document_id]
        )

    return dict(grouped)


def select_answerable_questions(
    queries_by_document: dict[str, list[str]],
    rng: random.Random,
) -> tuple[list[tuple[str, str]], list[str]]:
    """
    Selects exactly 300 answerable query/document pairs.

    A maximum of six questions is selected per positive document.
    Documents with more available questions are considered first.
    """
    document_candidates = sorted(
        queries_by_document.items(),
        key=lambda item: (
            -len(item[1]),
            item[0],
        ),
    )

    selected_pairs: list[tuple[str, str]] = []
    selected_document_ids: list[str] = []

    for document_id, available_query_ids in document_candidates:
        remaining = (
            RETRIEVAL_ANSWERABLE_COUNT
            - len(selected_pairs)
        )

        if remaining <= 0:
            break

        number_to_select = min(
            MAX_QUESTIONS_PER_POSITIVE_DOCUMENT,
            len(available_query_ids),
            remaining,
        )

        if number_to_select <= 0:
            continue

        selected_query_ids = sorted(
            rng.sample(
                available_query_ids,
                number_to_select,
            )
        )

        selected_document_ids.append(document_id)

        selected_pairs.extend(
            (query_id, document_id)
            for query_id in selected_query_ids
        )

    if len(selected_pairs) != RETRIEVAL_ANSWERABLE_COUNT:
        raise ValueError(
            "Could not create the requested number of answerable "
            f"questions. Required: {RETRIEVAL_ANSWERABLE_COUNT}; "
            f"selected: {len(selected_pairs)}."
        )

    return (
        selected_pairs,
        sorted(set(selected_document_ids)),
    )


def select_distractor_documents(
    pdf_urls: dict[str, Any],
    referenced_document_ids: set[str],
    positive_document_ids: set[str],
    rng: random.Random,
) -> list[str]:
    """
    Selects PDFs that are never a positive target in qrels.

    If the dataset does not contain enough completely unreferenced PDFs,
    the function falls back to referenced documents that are not positive
    benchmark sources.
    """
    unreferenced_candidates = sorted(
        set(pdf_urls.keys())
        - referenced_document_ids
        - positive_document_ids
    )

    selected: list[str] = []

    if unreferenced_candidates:
        number_from_unreferenced = min(
            NUM_DISTRACTOR_DOCUMENTS,
            len(unreferenced_candidates),
        )

        selected.extend(
            rng.sample(
                unreferenced_candidates,
                number_from_unreferenced,
            )
        )

    remaining_needed = (
        NUM_DISTRACTOR_DOCUMENTS
        - len(selected)
    )

    if remaining_needed > 0:
        fallback_candidates = sorted(
            referenced_document_ids
            - positive_document_ids
            - set(selected)
        )

        if len(fallback_candidates) < remaining_needed:
            raise ValueError(
                "Not enough documents are available to select "
                f"{NUM_DISTRACTOR_DOCUMENTS} distractor PDFs."
            )

        selected.extend(
            rng.sample(
                fallback_candidates,
                remaining_needed,
            )
        )

    return sorted(selected)


def select_abstention_queries(
    qrels: dict[str, Any],
    indexed_document_ids: set[str],
    already_selected_query_ids: set[str],
    rng: random.Random,
) -> list[tuple[str, str]]:
    """
    Selects exactly 100 real questions whose true source is not indexed.
    """
    candidates: list[tuple[str, str]] = []

    for query_id, qrel_record in qrels.items():
        document_id = extract_document_id(
            qrel_record,
            query_id,
        )

        if query_id in already_selected_query_ids:
            continue

        if document_id in indexed_document_ids:
            continue

        candidates.append(
            (query_id, document_id)
        )

    candidates.sort(
        key=lambda pair: pair[0]
    )

    if len(candidates) < RETRIEVAL_ABSTENTION_COUNT:
        raise ValueError(
            "Not enough source-not-indexed questions are available. "
            f"Required: {RETRIEVAL_ABSTENTION_COUNT}; "
            f"available: {len(candidates)}."
        )

    return sorted(
        rng.sample(
            candidates,
            RETRIEVAL_ABSTENTION_COUNT,
        ),
        key=lambda pair: pair[0],
    )


# ==========================================================
# Entry creation
# ==========================================================

def build_answerable_entries(
    selected_pairs: list[tuple[str, str]],
    queries: dict[str, Any],
    answers: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Creates answerable benchmark records.
    """
    entries = []

    for position, (
        query_id,
        document_id,
    ) in enumerate(
        selected_pairs,
        start=1,
    ):
        entries.append(
            {
                "benchmark_id": (
                    f"answerable_{position:04d}"
                ),
                "query_id": query_id,
                "question": extract_question_text(
                    queries,
                    query_id,
                ),
                "expected_answer": extract_expected_answer(
                    answers,
                    query_id,
                ),
                "expected_source": (
                    f"{document_id}.pdf"
                ),
                "expected_page": None,
                "relevant_pages": [],
                "should_abstain": False,
                "category": "answerable",
                "difficulty": "unclassified",
                "source_document_id": document_id,
            }
        )

    return entries


def build_abstention_entries(
    selected_pairs: list[tuple[str, str]],
    queries: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Creates source-not-indexed abstention records.
    """
    entries = []

    for position, (
        query_id,
        excluded_document_id,
    ) in enumerate(
        selected_pairs,
        start=1,
    ):
        entries.append(
            {
                "benchmark_id": (
                    f"abstention_{position:04d}"
                ),
                "query_id": query_id,
                "question": extract_question_text(
                    queries,
                    query_id,
                ),
                "expected_answer": None,
                "expected_source": None,
                "expected_page": None,
                "relevant_pages": [],
                "should_abstain": True,
                "category": "source_not_indexed",
                "difficulty": "unclassified",
                # Stored for auditability only. This source is not indexed
                # and must not be treated as an expected retrieval result.
                "excluded_source_document_id": (
                    excluded_document_id
                ),
            }
        )

    return entries


def clone_entry(
    entry: dict[str, Any],
    benchmark_id: str,
) -> dict[str, Any]:
    """
    Copies an entry and assigns an ID for the full benchmark.
    """
    cloned = dict(entry)
    cloned["master_benchmark_id"] = entry["benchmark_id"]
    cloned["benchmark_id"] = benchmark_id
    return cloned


def build_full_benchmark_subset(
    answerable_entries: list[dict[str, Any]],
    abstention_entries: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """
    Selects the deterministic 120-question full-pipeline subset.
    """
    if len(answerable_entries) < FULL_ANSWERABLE_COUNT:
        raise ValueError(
            "Not enough answerable entries are available "
            "for the full benchmark."
        )

    if len(abstention_entries) < FULL_ABSTENTION_COUNT:
        raise ValueError(
            "Not enough abstention entries are available "
            "for the full benchmark."
        )

    selected_answerable = rng.sample(
        answerable_entries,
        FULL_ANSWERABLE_COUNT,
    )

    selected_abstention = rng.sample(
        abstention_entries,
        FULL_ABSTENTION_COUNT,
    )

    selected_answerable.sort(
        key=lambda entry: entry["query_id"]
    )

    selected_abstention.sort(
        key=lambda entry: entry["query_id"]
    )

    full_entries = []

    for position, entry in enumerate(
        selected_answerable,
        start=1,
    ):
        full_entries.append(
            clone_entry(
                entry,
                f"full_answerable_{position:04d}",
            )
        )

    for position, entry in enumerate(
        selected_abstention,
        start=1,
    ):
        full_entries.append(
            clone_entry(
                entry,
                f"full_abstention_{position:04d}",
            )
        )

    return full_entries


# ==========================================================
# Validation
# ==========================================================

def validate_unique_field(
    entries: list[dict[str, Any]],
    field_name: str,
    dataset_name: str,
) -> None:
    """
    Ensures that one field is unique across a benchmark.
    """
    values = [
        entry[field_name]
        for entry in entries
    ]

    if len(values) != len(set(values)):
        raise ValueError(
            f"{dataset_name} contains duplicate "
            f"{field_name} values."
        )


def validate_benchmark(
    entries: list[dict[str, Any]],
    expected_total: int,
    expected_answerable: int,
    expected_abstention: int,
    dataset_name: str,
) -> None:
    """
    Validates benchmark size, balance, and uniqueness.
    """
    if len(entries) != expected_total:
        raise ValueError(
            f"{dataset_name} should contain {expected_total} "
            f"questions, but contains {len(entries)}."
        )

    answerable_count = sum(
        not entry["should_abstain"]
        for entry in entries
    )

    abstention_count = sum(
        entry["should_abstain"]
        for entry in entries
    )

    if answerable_count != expected_answerable:
        raise ValueError(
            f"{dataset_name} should contain "
            f"{expected_answerable} answerable questions, "
            f"but contains {answerable_count}."
        )

    if abstention_count != expected_abstention:
        raise ValueError(
            f"{dataset_name} should contain "
            f"{expected_abstention} abstention questions, "
            f"but contains {abstention_count}."
        )

    validate_unique_field(
        entries,
        "benchmark_id",
        dataset_name,
    )

    validate_unique_field(
        entries,
        "query_id",
        dataset_name,
    )

    for entry in entries:
        if not entry.get("question", "").strip():
            raise ValueError(
                f"{dataset_name} contains an empty question."
            )

        if entry["should_abstain"]:
            if entry.get("expected_answer") is not None:
                raise ValueError(
                    f"{dataset_name} abstention entry "
                    "contains an expected answer."
                )

            if entry.get("expected_source") is not None:
                raise ValueError(
                    f"{dataset_name} abstention entry "
                    "contains an expected source."
                )
        else:
            if not entry.get("expected_answer"):
                raise ValueError(
                    f"{dataset_name} answerable entry "
                    "has no expected answer."
                )

            if not entry.get("expected_source"):
                raise ValueError(
                    f"{dataset_name} answerable entry "
                    "has no expected source."
                )


def validate_full_is_subset(
    retrieval_entries: list[dict[str, Any]],
    full_entries: list[dict[str, Any]],
) -> None:
    """
    Ensures that every full-evaluation query belongs to the retrieval set.
    """
    retrieval_query_ids = {
        entry["query_id"]
        for entry in retrieval_entries
    }

    full_query_ids = {
        entry["query_id"]
        for entry in full_entries
    }

    if not full_query_ids <= retrieval_query_ids:
        raise ValueError(
            "The full benchmark is not a subset "
            "of the retrieval benchmark."
        )


# ==========================================================
# Main benchmark build
# ==========================================================

def main() -> None:
    rng = random.Random(RANDOM_SEED)

    print("=" * 72)
    print("CERTUS Open RAGBench benchmark builder")
    print("=" * 72)

    qrels = load_json_object(
        METADATA_DIR / "qrels.json"
    )

    queries = load_json_object(
        METADATA_DIR / "queries.json"
    )

    answers = load_json_object(
        METADATA_DIR / "answers.json"
    )

    pdf_urls = load_json_object(
        METADATA_DIR / "pdf_urls.json"
    )

    print("Validating Open RAGBench metadata...")

    validate_metadata(
        qrels=qrels,
        queries=queries,
        answers=answers,
        pdf_urls=pdf_urls,
    )

    queries_by_document = group_queries_by_document(
        qrels
    )

    selected_answerable_pairs, positive_document_ids = (
        select_answerable_questions(
            queries_by_document=queries_by_document,
            rng=rng,
        )
    )

    positive_document_id_set = set(
        positive_document_ids
    )

    referenced_document_ids = {
        extract_document_id(
            qrel_record,
            query_id,
        )
        for query_id, qrel_record in qrels.items()
    }

    distractor_document_ids = (
        select_distractor_documents(
            pdf_urls=pdf_urls,
            referenced_document_ids=referenced_document_ids,
            positive_document_ids=positive_document_id_set,
            rng=rng,
        )
    )

    indexed_document_ids = sorted(
        positive_document_id_set
        | set(distractor_document_ids)
    )

    selected_answerable_query_ids = {
        query_id
        for query_id, _ in selected_answerable_pairs
    }

    selected_abstention_pairs = (
        select_abstention_queries(
            qrels=qrels,
            indexed_document_ids=set(
                indexed_document_ids
            ),
            already_selected_query_ids=(
                selected_answerable_query_ids
            ),
            rng=rng,
        )
    )

    answerable_entries = build_answerable_entries(
        selected_pairs=selected_answerable_pairs,
        queries=queries,
        answers=answers,
    )

    abstention_entries = build_abstention_entries(
        selected_pairs=selected_abstention_pairs,
        queries=queries,
    )

    retrieval_entries = (
        answerable_entries
        + abstention_entries
    )

    full_entries = build_full_benchmark_subset(
        answerable_entries=answerable_entries,
        abstention_entries=abstention_entries,
        rng=rng,
    )

    validate_benchmark(
        entries=retrieval_entries,
        expected_total=400,
        expected_answerable=300,
        expected_abstention=100,
        dataset_name="Retrieval benchmark",
    )

    validate_benchmark(
        entries=full_entries,
        expected_total=120,
        expected_answerable=90,
        expected_abstention=30,
        dataset_name="Full benchmark",
    )

    validate_full_is_subset(
        retrieval_entries=retrieval_entries,
        full_entries=full_entries,
    )

    print("\nSelected benchmark composition:")
    print(
        f"  Positive PDFs:      "
        f"{len(positive_document_ids)}"
    )
    print(
        f"  Distractor PDFs:    "
        f"{len(distractor_document_ids)}"
    )
    print(
        f"  Total indexed PDFs: "
        f"{len(indexed_document_ids)}"
    )
    print(
        f"  Retrieval questions: "
        f"{len(retrieval_entries)}"
    )
    print(
        f"  Full questions:      "
        f"{len(full_entries)}"
    )

    PDF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nDownloading required PDFs...")

    for position, document_id in enumerate(
        indexed_document_ids,
        start=1,
    ):
        print(
            f"[{position}/{len(indexed_document_ids)}]",
            end=" ",
        )

        url = extract_pdf_url(
            pdf_urls,
            document_id,
        )

        download_pdf(
            document_id=document_id,
            url=url,
            output_path=(
                PDF_DIR / f"{document_id}.pdf"
            ),
        )

    missing_or_invalid_pdfs = [
        document_id
        for document_id in indexed_document_ids
        if not is_valid_pdf(
            PDF_DIR / f"{document_id}.pdf"
        )
    ]

    if missing_or_invalid_pdfs:
        raise RuntimeError(
            "Benchmark creation finished with missing or "
            "invalid PDFs: "
            + ", ".join(missing_or_invalid_pdfs)
        )

    manifest = {
        "benchmark_name": "certus_open_ragbench",
        "benchmark_version": "2.0",
        "random_seed": RANDOM_SEED,
        "retrieval_benchmark": {
            "file": str(
                RETRIEVAL_BENCHMARK_FILE.relative_to(
                    PROJECT_ROOT
                )
            ),
            "num_questions": len(
                retrieval_entries
            ),
            "num_answerable_questions": (
                RETRIEVAL_ANSWERABLE_COUNT
            ),
            "num_abstention_questions": (
                RETRIEVAL_ABSTENTION_COUNT
            ),
        },
        "full_benchmark": {
            "file": str(
                FULL_BENCHMARK_FILE.relative_to(
                    PROJECT_ROOT
                )
            ),
            "num_questions": len(full_entries),
            "num_answerable_questions": (
                FULL_ANSWERABLE_COUNT
            ),
            "num_abstention_questions": (
                FULL_ABSTENTION_COUNT
            ),
        },
        "selection": {
            "max_questions_per_positive_document": (
                MAX_QUESTIONS_PER_POSITIVE_DOCUMENT
            ),
            "num_positive_documents": len(
                positive_document_ids
            ),
            "num_distractor_documents": len(
                distractor_document_ids
            ),
            "num_indexed_documents": len(
                indexed_document_ids
            ),
        },
        "positive_document_ids": (
            positive_document_ids
        ),
        "distractor_document_ids": (
            distractor_document_ids
        ),
        "indexed_document_ids": (
            indexed_document_ids
        ),
        "pdf_directory": str(
            PDF_DIR.relative_to(PROJECT_ROOT)
        ),
        "ground_truth_scope": {
            "retrieval_relevance": "document-level",
            "page_level_labels_available": False,
            "note": (
                "Open RAGBench metadata used here provides "
                "document-level source relevance. expected_page "
                "therefore remains null until a manually annotated "
                "page-level subset is created."
            ),
        },
    }

    write_json(
        RETRIEVAL_BENCHMARK_FILE,
        retrieval_entries,
    )

    write_json(
        FULL_BENCHMARK_FILE,
        full_entries,
    )

    # Keep existing evaluation scripts working temporarily.
    write_json(
        COMPATIBILITY_BENCHMARK_FILE,
        full_entries,
    )

    write_json(
        MANIFEST_FILE,
        manifest,
    )

    print("\n" + "=" * 72)
    print("Benchmark creation completed successfully")
    print("=" * 72)
    print(
        f"Retrieval benchmark: "
        f"{RETRIEVAL_BENCHMARK_FILE}"
    )
    print(
        f"  Questions: 400 "
        f"(300 answerable, 100 abstention)"
    )
    print(
        f"Full benchmark:      "
        f"{FULL_BENCHMARK_FILE}"
    )
    print(
        f"  Questions: 120 "
        f"(90 answerable, 30 abstention)"
    )
    print(
        f"Compatibility file: "
        f"{COMPATIBILITY_BENCHMARK_FILE}"
    )
    print(
        f"Manifest:           "
        f"{MANIFEST_FILE}"
    )
    print(
        f"Indexed PDFs:       "
        f"{len(indexed_document_ids)}"
    )


if __name__ == "__main__":
    main()