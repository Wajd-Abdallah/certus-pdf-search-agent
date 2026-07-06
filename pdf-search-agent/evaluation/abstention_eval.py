"""
Evaluates abstention behavior using the FULL pipeline (retrieval + generation),
not just retrieval. For each benchmark question, checks whether the system
abstained exactly when it should have.

Run from the project root:
    python3 -m evaluation.abstention_eval
"""
import evaluation._env
import json
from pathlib import Path

from evaluation.benchmark_loader import load_small_benchmark

from app.pipeline import answer_question


def classify_outcome(should_abstain: bool, did_abstain: bool, abstention_reason: str | None) -> str:
    if abstention_reason == "generation_error":
        return "system_error"  # not a real evidence-based judgment -- Ollama/infra failed

    if should_abstain and did_abstain:
        return "true_positive"       # correctly abstained
    if should_abstain and not did_abstain:
        return "false_negative"      # should've abstained, didn't -- RISK: possible hallucination
    if not should_abstain and not did_abstain:
        return "true_negative"       # correctly answered
    return "false_positive"          # should've answered, abstained instead -- overcautious


def main():
    benchmark = load_small_benchmark()

    outputs = []
    counts = {
        "true_positive": 0,
        "true_negative": 0,
        "false_negative": 0,
        "false_positive": 0,
        "system_error": 0,
    }
    errors = 0

    for sample in benchmark:
        print("=" * 60)
        print("Question:", sample.question)
        print("Should abstain:", sample.should_abstain)

        try:
            result = answer_question(sample.question)
        except Exception as e:
            print("Pipeline error:", e)
            errors += 1
            outputs.append({
                "question": sample.question,
                "should_abstain": sample.should_abstain,
                "did_abstain": None,
                "outcome": "error",
                "error": str(e),
            })
            continue

        did_abstain = bool(result.get("abstained", False))
        abstention_reason = result.get("abstention_reason")
        outcome = classify_outcome(sample.should_abstain, did_abstain, abstention_reason)
        counts[outcome] += 1

        print("Did abstain:", did_abstain)
        print("Abstention reason:", abstention_reason)
        print("Outcome:", outcome)

        outputs.append({
            "question": sample.question,
            "should_abstain": sample.should_abstain,
            "did_abstain": did_abstain,
            "abstention_reason": abstention_reason,
            "outcome": outcome,
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
        })

    evaluated = (
        counts["true_positive"]
        + counts["true_negative"]
        + counts["false_negative"]
        + counts["false_positive"]
    )
    correct = counts["true_positive"] + counts["true_negative"]
    accuracy = correct / evaluated if evaluated else 0.0

    summary = {
        "num_questions": len(benchmark),
        "num_evaluated": evaluated,
        "num_pipeline_errors": errors,
        "num_system_errors": counts["system_error"],
        "abstention_accuracy": accuracy,
        **counts,
    }

    print("\n" + "=" * 60)
    print("Abstention Evaluation Summary")
    print("=" * 60)
    print(f"Accuracy (of {evaluated} evaluated): {accuracy:.3f}")
    print(f"True Positives  (correctly abstained):              {counts['true_positive']}")
    print(f"True Negatives  (correctly answered):               {counts['true_negative']}")
    print(f"False Negatives (should abstain, didn't -- RISK):   {counts['false_negative']}")
    print(f"False Positives (should answer, abstained instead): {counts['false_positive']}")
    if counts["system_error"]:
        print(f"System errors (excluded from accuracy):             {counts['system_error']}")
    if errors:
        print(f"Pipeline errors (question failed to run):           {errors}")

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "abstention_outputs.json").open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    with (output_dir / "abstention_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_dir / 'abstention_outputs.json'}")
    print(f"Summary saved to: {output_dir / 'abstention_summary.json'}")


if __name__ == "__main__":
    main()