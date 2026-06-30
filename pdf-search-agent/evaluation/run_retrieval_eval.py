import json
from pathlib import Path

from evaluation.benchmark_loader import load_small_benchmark
from evaluation.retrieval_metrics import evaluate_question

try:
    from app.indexer import Indexer
    from app.retriever import Retriever
except ModuleNotFoundError:
    from indexer import Indexer
    from retriever import Retriever


def main():
    benchmark = load_small_benchmark()

    indexer = Indexer()
    retriever = Retriever(indexer)

    outputs = []

    for sample in benchmark:
        print("=" * 60)
        print("Question:", sample.question)

        try:
            results = retriever.retrieve(sample.question)
            print(f"Retrieved {len(results)} chunks")

            clean_chunks = [
                {
                    "text": chunk.get("text", ""),
                    "source": chunk.get("source", chunk.get("document_name", "")),
                    "page": chunk.get("page", chunk.get("page_number", None)),
                    "score": chunk.get("score", chunk.get("similarity", None)),
                }
                for chunk in results
            ]

            metrics = evaluate_question(
                clean_chunks,
                sample.expected_source,
                sample.expected_page,
            )

            print("Metrics:")
            for name, value in metrics.items():
                print(f"  {name}: {value:.3f}")

            outputs.append({
                "question": sample.question,
                "expected_answer": sample.expected_answer,
                "expected_source": sample.expected_source,
                "expected_page": sample.expected_page,
                "retrieved_chunks": clean_chunks,
                "metrics": metrics,
            })

        except Exception as e:
            print("Retrieval could not run:")
            print(e)

            outputs.append({
                "question": sample.question,
                "expected_answer": sample.expected_answer,
                "expected_source": sample.expected_source,
                "expected_page": sample.expected_page,
                "retrieved_chunks": [],
                "metrics": {
                    "Recall@k": 0.0,
                    "MRR": 0.0,
                    "nDCG@k": 0.0,
                },
                "error": str(e),
            })

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "retrieval_outputs.json"
    summary_file = output_dir / "retrieval_summary.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    metric_names = ["Recall@k", "MRR", "nDCG@k"]
    summary = {}

    for metric_name in metric_names:
        values = [item["metrics"][metric_name] for item in outputs]
        summary[metric_name] = sum(values) / len(values) if values else 0.0

    summary["num_questions"] = len(outputs)

    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nEvaluation finished successfully.")
    print(f"Detailed results saved to: {output_file}")
    print(f"Metric summary saved to: {summary_file}")


if __name__ == "__main__":
    main()