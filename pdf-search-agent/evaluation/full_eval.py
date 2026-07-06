"""
Comprehensive evaluation: runs the FULL pipeline once per question and
computes abstention accuracy, citation correctness, Exact Match, F1,
LLM-as-a-judge score, a lexical grounding proxy, token usage, and
latency, in a single pass.

Metric functions (Exact Match, F1, citation correctness, lexical
grounding, cost estimation, abstention classification) live in
evaluation/answer_metrics.py so they can be unit-tested in isolation
without importing app.pipeline or Ollama.

Run from the project root:
    python3 -m evaluation.full_eval
"""

import evaluation._env  # sets PDF_AGENT_CHROMA_DIR before anything else -- must be first
import json
import re
import time
from pathlib import Path

import ollama

from evaluation.benchmark_loader import load_small_benchmark
from evaluation.answer_metrics import (
    compute_exact_match,
    compute_f1,
    citation_correct,
    lexical_grounding_score,
    estimate_illustrative_cost_usd,
    classify_abstention,
    ILLUSTRATIVE_USD_PER_1K_INPUT_TOKENS,
    ILLUSTRATIVE_USD_PER_1K_OUTPUT_TOKENS,
)
from app.pipeline import answer_question


JUDGE_PROMPT_TEMPLATE = """You are evaluating the quality of an AI-generated answer against a reference answer.

Question: {question}

Reference answer (ground truth): {reference}

AI-generated answer: {answer}

Rate how well the AI-generated answer captures the same information as the reference answer, on a scale from 1 to 5:
1 = completely wrong or unrelated
2 = mostly wrong, minor overlap
3 = partially correct, missing key information
4 = mostly correct, minor omissions
5 = fully correct and equivalent in meaning

Respond with ONLY a single digit (1, 2, 3, 4, or 5). Do not explain."""


def llm_judge_score(question: str, answer: str, reference: str) -> int | None:
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, reference=reference, answer=answer)
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        text = response["message"]["content"].strip()
        match = re.search(r"[1-5]", text)
        return int(match.group()) if match else None
    except Exception as e:
        print("Judge call failed:", e)
        return None


def main():
    benchmark = load_small_benchmark()

    outputs = []
    abstain_counts = {
        "true_positive": 0, "true_negative": 0,
        "false_negative": 0, "false_positive": 0, "system_error": 0,
    }
    em_scores = []
    f1_scores = []
    judge_scores = []
    citation_results = []
    grounding_scores = []
    latencies = []
    total_tokens_list = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for sample in benchmark:
        print("=" * 60)
        print("Question:", sample.question)

        start = time.time()
        try:
            result = answer_question(sample.question)
        except Exception as e:
            print("Pipeline error:", e)
            outputs.append({"question": sample.question, "error": str(e)})
            continue
        latency = time.time() - start
        latencies.append(latency)

        did_abstain = bool(result.get("abstained", False))
        abstention_reason = result.get("abstention_reason")
        abstain_outcome = classify_abstention(sample.should_abstain, did_abstain, abstention_reason)
        abstain_counts[abstain_outcome] += 1

        prompt_tokens = result.get("prompt_tokens")
        completion_tokens = result.get("completion_tokens")
        total_tokens = None
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
            total_tokens_list.append(total_tokens)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

        entry = {
            "question": sample.question,
            "should_abstain": sample.should_abstain,
            "did_abstain": did_abstain,
            "abstention_outcome": abstain_outcome,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "latency_seconds": round(latency, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        # Lexical grounding: computed for any real (non-abstained) answer,
        # regardless of whether reference ground truth exists -- it only
        # needs the generated answer and the retrieved context text.
        if not did_abstain:
            grounding = lexical_grounding_score(
                result.get("answer", ""),
                result.get("retrieved_contexts", []),
            )
            grounding_scores.append(grounding)
            entry["lexical_grounding_score"] = round(grounding, 3)

        if not sample.should_abstain and sample.expected_answer and not did_abstain:
            answer_text = result.get("answer", "")

            em = compute_exact_match(answer_text, sample.expected_answer)
            f1 = compute_f1(answer_text, sample.expected_answer)
            em_scores.append(em)
            f1_scores.append(f1)
            entry["exact_match"] = em
            entry["f1"] = round(f1, 3)

            correct = citation_correct(result.get("citations", []), sample.expected_source)
            citation_results.append(correct)
            entry["citation_correct"] = correct

            judge = llm_judge_score(sample.question, answer_text, sample.expected_answer)
            if judge is not None:
                judge_scores.append(judge)
            entry["llm_judge_score"] = judge

        print(f"Latency: {latency:.2f}s | Tokens: {total_tokens} | Abstained: {did_abstain} | Outcome: {abstain_outcome}")
        if "lexical_grounding_score" in entry:
            print(f"Lexical grounding score: {entry['lexical_grounding_score']}")
        if "f1" in entry:
            print(f"EM: {entry['exact_match']:.1f} | F1: {entry['f1']:.3f} | "
                  f"Citation correct: {entry['citation_correct']} | Judge: {entry['llm_judge_score']}/5")

        outputs.append(entry)

    evaluated = sum(v for k, v in abstain_counts.items() if k != "system_error")
    correct_abstain = abstain_counts["true_positive"] + abstain_counts["true_negative"]

    total_tokens_used = sum(total_tokens_list) if total_tokens_list else 0
    illustrative_cost = estimate_illustrative_cost_usd(total_prompt_tokens, total_completion_tokens)

    summary = {
        "num_questions": len(benchmark),
        "abstention_accuracy": correct_abstain / evaluated if evaluated else 0.0,
        **abstain_counts,
        "avg_exact_match": sum(em_scores) / len(em_scores) if em_scores else None,
        "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else None,
        "avg_llm_judge_score": sum(judge_scores) / len(judge_scores) if judge_scores else None,
        "avg_lexical_grounding_score": (
            sum(grounding_scores) / len(grounding_scores) if grounding_scores else None
        ),
        "citation_accuracy": sum(citation_results) / len(citation_results) if citation_results else None,
        "avg_latency_seconds": sum(latencies) / len(latencies) if latencies else None,
        "avg_total_tokens_per_question": (
            sum(total_tokens_list) / len(total_tokens_list) if total_tokens_list else None
        ),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens_used_all_questions": total_tokens_used,
        "actual_cost_usd": 0.0,
        "illustrative_cost_usd_if_hosted_api": round(illustrative_cost, 4),
        "note": (
            f"Actual cost is $0 (local Ollama inference). The illustrative cost "
            f"models input tokens at ${ILLUSTRATIVE_USD_PER_1K_INPUT_TOKENS}/1K and "
            f"output tokens at ${ILLUSTRATIVE_USD_PER_1K_OUTPUT_TOKENS}/1K -- separate "
            f"rates, since output tokens are typically priced higher by hosted "
            f"providers -- purely for comparison against a hypothetical hosted API. "
            f"avg_lexical_grounding_score is a weak proxy measuring vocabulary "
            f"overlap between the answer and retrieved context -- it is NOT "
            f"a faithfulness or correctness check, and should be interpreted "
            f"only as a lightweight signal, not proof of grounding."
        ),
        "num_scored_answers": len(em_scores),
        "num_lexical_grounding_scored": len(grounding_scores),
    }

    print("\n" + "=" * 60)
    print("Full Evaluation Summary")
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k}: {v}")

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "full_eval_outputs.json").open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    with (output_dir / "full_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to: {output_dir}/full_eval_outputs.json and full_eval_summary.json")


if __name__ == "__main__":
    main()
