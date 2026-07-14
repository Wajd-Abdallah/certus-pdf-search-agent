"""
Runs retrieval-only evaluation on the 400-question benchmark.

Benchmark composition:
- 300 answerable questions
- 100 source-not-indexed questions

Answerable questions are evaluated with retrieval metrics:
- Recall@k
- MRR
- nDCG@k

Source-not-indexed questions are recorded separately. They must not be treated
as ordinary retrieval failures because they intentionally have no expected
source in the indexed collection.

Run from the project root:

    python -m evaluation.run_retrieval_eval

For a small smoke test:

    python -m evaluation.run_retrieval_eval --max-questions 5

Evaluation isolation:
evaluation._env is imported before app modules, so this script uses the
dedicated evaluation ChromaDB directory instead of the live Streamlit data.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import evaluation._env  # must be imported before app.* modules

from app.config import load_config
from app.indexer import Indexer
from app.retriever import Retriever
from evaluation.benchmark_loader import load_retrieval_benchmark
from evaluation.retrieval_metrics import evaluate_question


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"

OUTPUT_FILE = OUTPUT_DIR / "retrieval_outputs.json"
SUMMARY_FILE = OUTPUT_DIR / "retrieval_summary.json"


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run retrieval evaluation on the CERTUS "
            "400-question benchmark."
        )
    )

    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help=(
            "Optional maximum number of benchmark questions. "
            "Useful for smoke tests."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Optional retrieval depth. "
            "Defaults to retrieval.top_k from baseline.yaml."
        ),
    )

    args = parser.parse_args()

    if (
        args.max_questions is not None
        and args.max_questions <= 0
    ):
        parser.error(
            "--max-questions must be a positive integer."
        )

    if args.top_k is not None and args.top_k <= 0:
        parser.error(
            "--top-k must be a positive integer."
        )

    return args


def mean(values: list[float]) -> float | None:
    """
    Returns the arithmetic mean or None for an empty list.
    """
    if not values:
        return None

    return sum(values) / len(values)


def percentile(
    values: list[float],
    percentile_value: float,
) -> float | None:
    """
    Computes a percentile using linear interpolation.
    """
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (
        percentile_value / 100.0
    ) * (len(ordered) - 1)

    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1,
    )

    fraction = position - lower_index

    return (
        ordered[lower_index]
        + (
            ordered[upper_index]
            - ordered[lower_index]
        )
        * fraction
    )


def write_json(
    path: Path,
    data: Any,
) -> None:
    """
    Writes JSON atomically.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def clean_retrieval_results(
    results: list[dict],
) -> list[dict]:
    """
    Converts Retriever output into an evaluation-friendly format.
    """
    clean_chunks = []

    for rank, chunk in enumerate(
        results,
        start=1,
    ):
        metadata = chunk.get(
            "metadata",
            {},
        ) or {}

        distance = chunk.get("distance")

        clean_chunks.append(
            {
                "rank": rank,
                "chunk_id": chunk.get("chunk_id"),
                "text": chunk.get(
                    "text_content",
                    "",
                ),
                "source": metadata.get(
                    "document_name",
                    "",
                ),
                "page": metadata.get(
                    "page_number"
                ),
                # Chroma distance: lower means more similar.
                "distance": distance,
            }
        )

    return clean_chunks


def evaluate_answerable_sample(
    clean_chunks: list[dict],
    expected_source: str,
    expected_page: int | None,
    top_k: int,
) -> dict[str, float]:
    """
    Computes retrieval metrics for an answerable question.
    """
    return evaluate_question(
        retrieved_chunks=clean_chunks,
        expected_source=expected_source,
        expected_page=expected_page,
        k=top_k,
    )


def build_summary(
    outputs: list[dict],
    top_k: int,
    collection_count: int,
    persist_directory: str,
    collection_name: str,
) -> dict:
    """
    Aggregates retrieval results.

    Positive retrieval metrics are calculated only over answerable questions.
    Source-not-indexed questions are summarized separately.
    """
    answerable_outputs = [
        item
        for item in outputs
        if not item["should_abstain"]
        and not item.get("error")
    ]

    abstention_outputs = [
        item
        for item in outputs
        if item["should_abstain"]
        and not item.get("error")
    ]

    error_outputs = [
        item
        for item in outputs
        if item.get("error")
    ]

    recall_values = [
        item["metrics"]["Recall@k"]
        for item in answerable_outputs
    ]

    mrr_values = [
        item["metrics"]["MRR"]
        for item in answerable_outputs
    ]

    ndcg_values = [
        item["metrics"]["nDCG@k"]
        for item in answerable_outputs
    ]

    all_latencies = [
        item["retrieval_latency_seconds"]
        for item in outputs
        if item.get(
            "retrieval_latency_seconds"
        ) is not None
    ]

    answerable_latencies = [
        item["retrieval_latency_seconds"]
        for item in answerable_outputs
    ]

    abstention_latencies = [
        item["retrieval_latency_seconds"]
        for item in abstention_outputs
    ]

    negative_best_distances = [
        item["best_distance"]
        for item in abstention_outputs
        if item.get("best_distance") is not None
    ]

    positive_best_distances = [
        item["best_distance"]
        for item in answerable_outputs
        if item.get("best_distance") is not None
    ]

    return {
        "benchmark": "benchmark_retrieval_400.json",
        "num_questions": len(outputs),
        "num_answerable_questions": len(
            [
                item
                for item in outputs
                if not item["should_abstain"]
            ]
        ),
        "num_source_not_indexed_questions": len(
            [
                item
                for item in outputs
                if item["should_abstain"]
            ]
        ),
        "num_answerable_questions_scored": len(
            answerable_outputs
        ),
        "num_source_not_indexed_questions_recorded": len(
            abstention_outputs
        ),
        "num_errors": len(error_outputs),
        "top_k": top_k,
        "answerable_retrieval_metrics": {
            "Recall@k": (
                mean(recall_values) or 0.0
            ),
            "MRR": (
                mean(mrr_values) or 0.0
            ),
            "nDCG@k": (
                mean(ndcg_values) or 0.0
            ),
        },
        "retrieval_latency_seconds": {
            "average_all": mean(
                all_latencies
            ),
            "average_answerable": mean(
                answerable_latencies
            ),
            "average_source_not_indexed": mean(
                abstention_latencies
            ),
            "p50_all": percentile(
                all_latencies,
                50,
            ),
            "p95_all": percentile(
                all_latencies,
                95,
            ),
            "maximum_all": (
                max(all_latencies)
                if all_latencies
                else None
            ),
        },
        "distance_analysis": {
            "note": (
                "Chroma distance is used: lower values indicate "
                "greater similarity. Source-not-indexed questions "
                "are not included in Recall/MRR/nDCG. Their best "
                "retrieval distances are recorded for later "
                "abstention-threshold calibration."
            ),
            "answerable_best_distance_average": mean(
                positive_best_distances
            ),
            "answerable_best_distance_p95": percentile(
                positive_best_distances,
                95,
            ),
            "source_not_indexed_best_distance_average": mean(
                negative_best_distances
            ),
            "source_not_indexed_best_distance_p05": percentile(
                negative_best_distances,
                5,
            ),
            "source_not_indexed_best_distance_minimum": (
                min(negative_best_distances)
                if negative_best_distances
                else None
            ),
        },
        "index": {
            "collection_name": collection_name,
            "persist_directory": persist_directory,
            "stored_chunks": collection_count,
        },
        "ground_truth_scope": {
            "retrieval_relevance": "document-level",
            "page_level_evaluation_enabled": False,
            "note": (
                "The Open RAGBench metadata used by this project "
                "does not provide verified page-level labels. "
                "The retrieval metrics therefore check the expected "
                "document, not a manually verified page."
            ),
        },
    }


def main() -> None:
    args = parse_args()

    config = load_config()

    persist_directory = (
        os.environ.get(
            "PDF_AGENT_CHROMA_DIR"
        )
        or config["index"][
            "persist_directory"
        ]
    )

    top_k = (
        args.top_k
        if args.top_k is not None
        else config["retrieval"]["top_k"]
    )

    benchmark = load_retrieval_benchmark()

    if args.max_questions is not None:
        benchmark = benchmark[
            :args.max_questions
        ]

    indexer = Indexer(
        collection_name=config["index"][
            "collection_name"
        ],
        persist_directory=persist_directory,
    )

    collection_count = (
        indexer.collection.count()
    )

    if collection_count <= 0:
        raise RuntimeError(
            "The evaluation ChromaDB collection is empty. "
            "Run `python index_benchmark_pdfs.py` first."
        )

    retriever = Retriever(
        indexer=indexer
    )

    print("=" * 72)
    print("CERTUS retrieval evaluation")
    print("=" * 72)
    print(
        f"Questions:          {len(benchmark)}"
    )
    print(
        f"Top-k:              {top_k}"
    )
    print(
        f"Collection:         {indexer.collection_name}"
    )
    print(
        f"Stored chunks:      {collection_count}"
    )
    print(
        f"Chroma directory:   {persist_directory}"
    )

    outputs: list[dict] = []

    total_questions = len(benchmark)

    for position, sample in enumerate(
        benchmark,
        start=1,
    ):
        print("\n" + "=" * 72)
        print(
            f"[{position}/{total_questions}] "
            f"{sample.question}"
        )
        print(
            f"Should abstain: {sample.should_abstain}"
        )

        start_time = time.perf_counter()

        try:
            results = retriever.retrieve(
                query=sample.question,
                top_k=top_k,
            )

            latency = (
                time.perf_counter()
                - start_time
            )

            clean_chunks = (
                clean_retrieval_results(
                    results
                )
            )

            best_distance = (
                clean_chunks[0]["distance"]
                if clean_chunks
                else None
            )

            entry = {
                "question_number": position,
                "question": sample.question,
                "should_abstain": (
                    sample.should_abstain
                ),
                "expected_answer": (
                    sample.expected_answer
                ),
                "expected_source": (
                    sample.expected_source
                ),
                "expected_page": (
                    sample.expected_page
                ),
                "retrieved_chunks": clean_chunks,
                "num_retrieved_chunks": len(
                    clean_chunks
                ),
                "best_distance": best_distance,
                "retrieval_latency_seconds": round(
                    latency,
                    6,
                ),
                "metrics": None,
                "error": None,
            }

            if sample.should_abstain:
                print(
                    "Source-not-indexed question: "
                    "positive retrieval metrics skipped."
                )

                if best_distance is not None:
                    print(
                        "Best retrieved distance: "
                        f"{best_distance:.4f}"
                    )
            else:
                if not sample.expected_source:
                    raise ValueError(
                        "Answerable benchmark question "
                        "has no expected source."
                    )

                metrics = (
                    evaluate_answerable_sample(
                        clean_chunks=clean_chunks,
                        expected_source=(
                            sample.expected_source
                        ),
                        expected_page=(
                            sample.expected_page
                        ),
                        top_k=top_k,
                    )
                )

                entry["metrics"] = metrics

                print("Metrics:")

                for metric_name, value in (
                    metrics.items()
                ):
                    print(
                        f"  {metric_name}: "
                        f"{value:.3f}"
                    )

            print(
                "Retrieval latency: "
                f"{latency:.4f}s"
            )

            outputs.append(entry)

        except Exception as error:
            latency = (
                time.perf_counter()
                - start_time
            )

            print(
                f"Retrieval error: {error}"
            )

            outputs.append(
                {
                    "question_number": position,
                    "question": sample.question,
                    "should_abstain": (
                        sample.should_abstain
                    ),
                    "expected_answer": (
                        sample.expected_answer
                    ),
                    "expected_source": (
                        sample.expected_source
                    ),
                    "expected_page": (
                        sample.expected_page
                    ),
                    "retrieved_chunks": [],
                    "num_retrieved_chunks": 0,
                    "best_distance": None,
                    "retrieval_latency_seconds": round(
                        latency,
                        6,
                    ),
                    "metrics": None,
                    "error": str(error),
                }
            )

    summary = build_summary(
        outputs=outputs,
        top_k=top_k,
        collection_count=collection_count,
        persist_directory=persist_directory,
        collection_name=indexer.collection_name,
    )

    write_json(
        OUTPUT_FILE,
        outputs,
    )

    write_json(
        SUMMARY_FILE,
        summary,
    )

    print("\n" + "=" * 72)
    print("Retrieval evaluation completed")
    print("=" * 72)

    metrics = summary[
        "answerable_retrieval_metrics"
    ]

    print(
        f"Questions processed: "
        f"{summary['num_questions']}"
    )
    print(
        f"Answerable scored:   "
        f"{summary['num_answerable_questions_scored']}"
    )
    print(
        f"Negative recorded:   "
        f"{summary['num_source_not_indexed_questions_recorded']}"
    )
    print(
        f"Errors:              "
        f"{summary['num_errors']}"
    )
    print(
        f"Recall@{top_k}:          "
        f"{metrics['Recall@k']:.3f}"
    )
    print(
        f"MRR:                 "
        f"{metrics['MRR']:.3f}"
    )
    print(
        f"nDCG@{top_k}:            "
        f"{metrics['nDCG@k']:.3f}"
    )
    print(
        f"Detailed results:    {OUTPUT_FILE}"
    )
    print(
        f"Summary:             {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()