"""
Runs the complete CERTUS RAG evaluation.

The evaluation includes:
- retrieval and answer generation,
- abstention classification,
- Exact Match,
- token-level F1,
- citation source correctness,
- lexical grounding proxy,
- LLM-as-a-judge,
- token usage,
- end-to-end latency.

Benchmark:
- 120 questions
- 90 answerable
- 30 source-not-indexed abstention questions

Run the full evaluation:

    python -m evaluation.full_eval

Run a small smoke test:

    python -m evaluation.full_eval --max-questions 5

Restart from the beginning:

    python -m evaluation.full_eval --restart

The script saves progress after every question. If a long run is interrupted,
running it again resumes from the saved checkpoint unless --restart is used.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import evaluation._env  # must be imported before app.pipeline

import ollama

from app.pipeline import answer_question
from evaluation.answer_metrics import (
    ILLUSTRATIVE_USD_PER_1K_INPUT_TOKENS,
    ILLUSTRATIVE_USD_PER_1K_OUTPUT_TOKENS,
    citation_correct,
    classify_abstention,
    compute_exact_match,
    compute_f1,
    estimate_illustrative_cost_usd,
    lexical_grounding_score,
)
from evaluation.benchmark_loader import load_full_benchmark


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"

OUTPUT_FILE = OUTPUT_DIR / "full_eval_outputs.json"
SUMMARY_FILE = OUTPUT_DIR / "full_eval_summary.json"
CHECKPOINT_FILE = OUTPUT_DIR / "full_eval_checkpoint.json"

JUDGE_MODEL = "llama3.2"


JUDGE_PROMPT_TEMPLATE = """You are evaluating the quality of an AI-generated answer against a reference answer.

Question:
{question}

Reference answer:
{reference}

AI-generated answer:
{answer}

Rate how well the AI-generated answer captures the same information as the reference answer.

Use this scale:
1 = completely wrong or unrelated
2 = mostly wrong, with only minor overlap
3 = partially correct, but missing important information
4 = mostly correct, with only minor omissions
5 = fully correct and equivalent in meaning

Respond with exactly one digit: 1, 2, 3, 4, or 5.
Do not provide an explanation."""


def parse_args() -> argparse.Namespace:
    """
    Parses command-line options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete CERTUS RAG evaluation "
            "on the 120-question benchmark."
        )
    )

    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N questions. "
            "Useful for smoke tests."
        ),
    )

    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Ignore and remove any existing checkpoint, "
            "then start from question one."
        ),
    )

    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help=(
            "Skip LLM-as-a-judge calls. Useful for faster "
            "development tests."
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

    return args


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


def load_json(path: Path) -> Any:
    """
    Loads JSON from disk.
    """
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def mean(
    values: list[float],
) -> float | None:
    """
    Returns the arithmetic mean, or None for an empty list.
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


def llm_judge_score(
    question: str,
    answer: str,
    reference: str,
) -> int | None:
    """
    Scores answer quality from 1 to 5 using the local Ollama model.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        reference=reference,
        answer=answer,
    )

    try:
        response = ollama.chat(
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0,
            },
        )

        response_text = (
            response["message"]["content"]
            .strip()
        )

        # The judge was instructed to return one digit.
        # fullmatch avoids accidentally accepting unrelated numbers.
        direct_match = re.fullmatch(
            r"\s*([1-5])\s*",
            response_text,
        )

        if direct_match:
            return int(
                direct_match.group(1)
            )

        # Defensive fallback for local models that add a short label.
        fallback_match = re.search(
            r"(?<!\d)([1-5])(?!\d)",
            response_text,
        )

        if fallback_match:
            return int(
                fallback_match.group(1)
            )

        print(
            "Judge returned an invalid response: "
            f"{response_text!r}"
        )

        return None

    except Exception as error:
        print(
            f"Judge call failed: {error}"
        )
        return None


def load_checkpoint(
    expected_questions: int,
) -> list[dict]:
    """
    Loads a valid checkpoint if one exists.
    """
    if not CHECKPOINT_FILE.exists():
        return []

    try:
        checkpoint = load_json(
            CHECKPOINT_FILE
        )
    except Exception as error:
        print(
            "Could not read checkpoint. "
            f"Starting again: {error}"
        )
        return []

    if not isinstance(checkpoint, dict):
        print(
            "Checkpoint has an invalid format. "
            "Starting again."
        )
        return []

    if checkpoint.get(
        "expected_questions"
    ) != expected_questions:
        print(
            "Checkpoint belongs to a different "
            "benchmark size. Starting again."
        )
        return []

    outputs = checkpoint.get(
        "outputs",
        [],
    )

    if not isinstance(outputs, list):
        print(
            "Checkpoint outputs are invalid. "
            "Starting again."
        )
        return []

    return outputs


def save_checkpoint(
    outputs: list[dict],
    expected_questions: int,
    skip_judge: bool,
) -> None:
    """
    Saves progress after every processed question.
    """
    checkpoint = {
        "benchmark": (
            "benchmark_full_120.json"
        ),
        "expected_questions": (
            expected_questions
        ),
        "completed_questions": len(
            outputs
        ),
        "skip_judge": skip_judge,
        "outputs": outputs,
    }

    write_json(
        CHECKPOINT_FILE,
        checkpoint,
    )


def build_summary(
    outputs: list[dict],
    expected_questions: int,
    skip_judge: bool,
) -> dict:
    """
    Aggregates all full-pipeline metrics.
    """
    abstention_counts = {
        "true_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
        "false_positive": 0,
        "system_error": 0,
    }

    pipeline_errors = 0

    em_scores: list[float] = []
    f1_scores: list[float] = []
    judge_scores: list[int] = []
    citation_results: list[bool] = []
    grounding_scores: list[float] = []
    latencies: list[float] = []
    prompt_tokens_values: list[int] = []
    completion_tokens_values: list[int] = []
    total_tokens_values: list[int] = []

    for entry in outputs:
        if entry.get("pipeline_error"):
            pipeline_errors += 1
            continue

        outcome = entry.get(
            "abstention_outcome"
        )

        if outcome in abstention_counts:
            abstention_counts[outcome] += 1

        latency = entry.get(
            "latency_seconds"
        )

        if isinstance(
            latency,
            (int, float),
        ):
            latencies.append(
                float(latency)
            )

        prompt_tokens = entry.get(
            "prompt_tokens"
        )

        completion_tokens = entry.get(
            "completion_tokens"
        )

        total_tokens = entry.get(
            "total_tokens"
        )

        if isinstance(prompt_tokens, int):
            prompt_tokens_values.append(
                prompt_tokens
            )

        if isinstance(
            completion_tokens,
            int,
        ):
            completion_tokens_values.append(
                completion_tokens
            )

        if isinstance(total_tokens, int):
            total_tokens_values.append(
                total_tokens
            )

        exact_match = entry.get(
            "exact_match"
        )

        if isinstance(
            exact_match,
            (int, float),
        ):
            em_scores.append(
                float(exact_match)
            )

        f1 = entry.get("f1")

        if isinstance(
            f1,
            (int, float),
        ):
            f1_scores.append(
                float(f1)
            )

        judge_score = entry.get(
            "llm_judge_score"
        )

        if isinstance(judge_score, int):
            judge_scores.append(
                judge_score
            )

        citation_result = entry.get(
            "citation_correct"
        )

        if isinstance(
            citation_result,
            bool,
        ):
            citation_results.append(
                citation_result
            )

        grounding_score = entry.get(
            "lexical_grounding_score"
        )

        if isinstance(
            grounding_score,
            (int, float),
        ):
            grounding_scores.append(
                float(grounding_score)
            )

    evaluated_abstention_cases = sum(
        value
        for name, value
        in abstention_counts.items()
        if name != "system_error"
    )

    correct_abstention_cases = (
        abstention_counts["true_positive"]
        + abstention_counts["true_negative"]
    )

    total_prompt_tokens = sum(
        prompt_tokens_values
    )

    total_completion_tokens = sum(
        completion_tokens_values
    )

    illustrative_cost = (
        estimate_illustrative_cost_usd(
            total_prompt_tokens,
            total_completion_tokens,
        )
    )

    summary = {
        "benchmark": (
            "benchmark_full_120.json"
        ),
        "expected_num_questions": (
            expected_questions
        ),
        "num_questions_processed": len(
            outputs
        ),
        "num_pipeline_errors": (
            pipeline_errors
        ),
        "evaluation_complete": (
            len(outputs)
            == expected_questions
        ),
        "judge_enabled": (
            not skip_judge
        ),
        "abstention": {
            "num_evaluated": (
                evaluated_abstention_cases
            ),
            "accuracy": (
                correct_abstention_cases
                / evaluated_abstention_cases
                if evaluated_abstention_cases
                else None
            ),
            **abstention_counts,
        },
        "answer_quality": {
            "num_scored_answers": len(
                f1_scores
            ),
            "average_exact_match": mean(
                em_scores
            ),
            "average_f1": mean(
                f1_scores
            ),
            "average_llm_judge_score": mean(
                [
                    float(score)
                    for score in judge_scores
                ]
            ),
            "num_llm_judge_scores": len(
                judge_scores
            ),
        },
        "citations": {
            "num_scored_answers": len(
                citation_results
            ),
            "source_accuracy": (
                sum(citation_results)
                / len(citation_results)
                if citation_results
                else None
            ),
            "note": (
                "Citation source accuracy checks whether "
                "at least one structured citation matches "
                "the expected source document. Open RAGBench "
                "does not provide verified page-level labels."
            ),
        },
        "grounding": {
            "num_scored_answers": len(
                grounding_scores
            ),
            "average_lexical_grounding_score": mean(
                grounding_scores
            ),
            "note": (
                "Lexical grounding is only a vocabulary-overlap "
                "proxy. It is not proof of factual correctness "
                "or faithfulness."
            ),
        },
        "latency_seconds": {
            "average": mean(
                latencies
            ),
            "p50": percentile(
                latencies,
                50,
            ),
            "p95": percentile(
                latencies,
                95,
            ),
            "maximum": (
                max(latencies)
                if latencies
                else None
            ),
        },
        "tokens": {
            "num_questions_with_token_data": len(
                total_tokens_values
            ),
            "average_prompt_tokens": mean(
                [
                    float(value)
                    for value
                    in prompt_tokens_values
                ]
            ),
            "average_completion_tokens": mean(
                [
                    float(value)
                    for value
                    in completion_tokens_values
                ]
            ),
            "average_total_tokens_per_question": mean(
                [
                    float(value)
                    for value
                    in total_tokens_values
                ]
            ),
            "total_prompt_tokens": (
                total_prompt_tokens
            ),
            "total_completion_tokens": (
                total_completion_tokens
            ),
            "total_tokens": sum(
                total_tokens_values
            ),
        },
        "cost": {
            "actual_cost_usd": 0.0,
            "illustrative_cost_usd_if_hosted_api": round(
                illustrative_cost,
                6,
            ),
            "illustrative_input_rate_usd_per_1k_tokens": (
                ILLUSTRATIVE_USD_PER_1K_INPUT_TOKENS
            ),
            "illustrative_output_rate_usd_per_1k_tokens": (
                ILLUSTRATIVE_USD_PER_1K_OUTPUT_TOKENS
            ),
            "note": (
                "Actual inference uses local Ollama, so the "
                "measured monetary API cost is zero. The hosted "
                "cost is illustrative only."
            ),
        },
        "ground_truth_scope": {
            "answer_reference": (
                "Open RAGBench reference answer"
            ),
            "citation_relevance": (
                "document-level source"
            ),
            "page_level_labels_available": False,
        },
    }

    return summary


def evaluate_one_question(
    question_number: int,
    sample,
    skip_judge: bool,
) -> dict:
    """
    Runs and scores one complete pipeline question.
    """
    start_time = time.perf_counter()

    try:
        result = answer_question(
            sample.question
        )
    except Exception as error:
        latency = (
            time.perf_counter()
            - start_time
        )

        return {
            "question_number": (
                question_number
            ),
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
            "pipeline_error": str(error),
            "latency_seconds": round(
                latency,
                6,
            ),
        }

    latency = (
        time.perf_counter()
        - start_time
    )

    did_abstain = bool(
        result.get(
            "abstained",
            False,
        )
    )

    abstention_reason = result.get(
        "abstention_reason"
    )

    abstention_outcome = (
        classify_abstention(
            should_abstain=(
                sample.should_abstain
            ),
            did_abstain=did_abstain,
            abstention_reason=(
                abstention_reason
            ),
        )
    )

    answer_text = str(
        result.get(
            "answer",
            "",
        )
    )

    citations = result.get(
        "citations",
        [],
    )

    if not isinstance(citations, list):
        citations = []

    retrieved_contexts = result.get(
        "retrieved_contexts",
        [],
    )

    if not isinstance(
        retrieved_contexts,
        list,
    ):
        retrieved_contexts = []

    prompt_tokens = result.get(
        "prompt_tokens"
    )

    completion_tokens = result.get(
        "completion_tokens"
    )

    total_tokens = None

    if (
        isinstance(prompt_tokens, int)
        and isinstance(
            completion_tokens,
            int,
        )
    ):
        total_tokens = (
            prompt_tokens
            + completion_tokens
        )

    entry = {
        "question_number": (
            question_number
        ),
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
        "did_abstain": did_abstain,
        "abstention_reason": (
            abstention_reason
        ),
        "abstention_outcome": (
            abstention_outcome
        ),
        "answer": answer_text,
        "citations": citations,
        "num_citations": len(
            citations
        ),
        "latency_seconds": round(
            latency,
            6,
        ),
        "prompt_tokens": (
            prompt_tokens
        ),
        "completion_tokens": (
            completion_tokens
        ),
        "total_tokens": (
            total_tokens
        ),
        "pipeline_error": None,
    }

    if not did_abstain:
        grounding_score = (
            lexical_grounding_score(
                answer_text,
                retrieved_contexts,
            )
        )

        entry[
            "lexical_grounding_score"
        ] = round(
            grounding_score,
            6,
        )

    if (
        not sample.should_abstain
        and sample.expected_answer
        and not did_abstain
    ):
        exact_match = (
            compute_exact_match(
                answer_text,
                sample.expected_answer,
            )
        )

        f1_score = compute_f1(
            answer_text,
            sample.expected_answer,
        )

        entry["exact_match"] = (
            exact_match
        )

        entry["f1"] = round(
            f1_score,
            6,
        )

        if sample.expected_source:
            entry["citation_correct"] = (
                citation_correct(
                    citations,
                    sample.expected_source,
                )
            )
        else:
            entry["citation_correct"] = None

        if skip_judge:
            entry[
                "llm_judge_score"
            ] = None
        else:
            entry[
                "llm_judge_score"
            ] = llm_judge_score(
                question=sample.question,
                answer=answer_text,
                reference=(
                    sample.expected_answer
                ),
            )

    return entry


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    benchmark = load_full_benchmark()

    if args.max_questions is not None:
        benchmark = benchmark[
            :args.max_questions
        ]

    expected_questions = len(
        benchmark
    )

    if args.restart:
        CHECKPOINT_FILE.unlink(
            missing_ok=True
        )

    outputs = load_checkpoint(
        expected_questions=(
            expected_questions
        )
    )

    if len(outputs) > expected_questions:
        print(
            "Checkpoint contains more records "
            "than this run requires. Restarting."
        )
        outputs = []

    start_position = len(outputs) + 1

    print("=" * 72)
    print("CERTUS full RAG evaluation")
    print("=" * 72)
    print(
        f"Questions:        {expected_questions}"
    )
    print(
        f"Judge model:      "
        f"{'disabled' if args.skip_judge else JUDGE_MODEL}"
    )
    print(
        f"Resume position:  {start_position}"
    )
    print(
        f"Checkpoint:       {CHECKPOINT_FILE}"
    )

    if outputs:
        print(
            f"Loaded {len(outputs)} completed "
            "question(s) from checkpoint."
        )

    for position in range(
        start_position,
        expected_questions + 1,
    ):
        sample = benchmark[
            position - 1
        ]

        print("\n" + "=" * 72)
        print(
            f"[{position}/{expected_questions}] "
            f"{sample.question}"
        )
        print(
            f"Should abstain: "
            f"{sample.should_abstain}"
        )

        entry = evaluate_one_question(
            question_number=position,
            sample=sample,
            skip_judge=args.skip_judge,
        )

        outputs.append(entry)

        save_checkpoint(
            outputs=outputs,
            expected_questions=(
                expected_questions
            ),
            skip_judge=args.skip_judge,
        )

        if entry.get(
            "pipeline_error"
        ):
            print(
                "Pipeline error: "
                f"{entry['pipeline_error']}"
            )
            continue

        print(
            "Latency: "
            f"{entry['latency_seconds']:.2f}s"
        )
        print(
            "Abstained: "
            f"{entry['did_abstain']}"
        )
        print(
            "Outcome: "
            f"{entry['abstention_outcome']}"
        )
        print(
            "Tokens: "
            f"{entry['total_tokens']}"
        )

        if (
            "lexical_grounding_score"
            in entry
        ):
            print(
                "Lexical grounding: "
                f"{entry['lexical_grounding_score']:.3f}"
            )

        if "f1" in entry:
            print(
                "Exact Match: "
                f"{entry['exact_match']:.3f}"
            )
            print(
                "F1: "
                f"{entry['f1']:.3f}"
            )
            print(
                "Citation correct: "
                f"{entry['citation_correct']}"
            )
            print(
                "Judge: "
                f"{entry['llm_judge_score']}"
            )

    summary = build_summary(
        outputs=outputs,
        expected_questions=(
            expected_questions
        ),
        skip_judge=args.skip_judge,
    )

    write_json(
        OUTPUT_FILE,
        outputs,
    )

    write_json(
        SUMMARY_FILE,
        summary,
    )

    if (
        len(outputs)
        == expected_questions
    ):
        CHECKPOINT_FILE.unlink(
            missing_ok=True
        )

    print("\n" + "=" * 72)
    print("Full RAG evaluation completed")
    print("=" * 72)
    print(
        f"Questions processed: "
        f"{summary['num_questions_processed']}"
    )
    print(
        f"Pipeline errors:     "
        f"{summary['num_pipeline_errors']}"
    )
    print(
        "Abstention accuracy: "
        f"{summary['abstention']['accuracy']}"
    )
    print(
        "Average F1:          "
        f"{summary['answer_quality']['average_f1']}"
    )
    print(
        "Average judge score: "
        f"{summary['answer_quality']['average_llm_judge_score']}"
    )
    print(
        "Citation accuracy:   "
        f"{summary['citations']['source_accuracy']}"
    )
    print(
        "Average latency:     "
        f"{summary['latency_seconds']['average']}"
    )
    print(
        f"Detailed results:    {OUTPUT_FILE}"
    )
    print(
        f"Summary:             {SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()