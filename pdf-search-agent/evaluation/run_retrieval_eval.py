import json
from pathlib import Path

from evaluation.benchmark_loader import load_small_benchmark

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

            clean_chunks = []

            for chunk in results:
                clean_chunks.append({
                    "text": chunk.get("text", ""),
                    "source": chunk.get("source", chunk.get("document_name", "")),
                    "page": chunk.get("page", chunk.get("page_number", None)),
                    "score": chunk.get("score", chunk.get("similarity", None)),
                })

            outputs.append({
                "question": sample.question,
                "expected_answer": sample.expected_answer,
                "expected_source": sample.expected_source,
                "expected_page": sample.expected_page,
                "retrieved_chunks": clean_chunks,
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
                "error": str(e),
            })

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "retrieval_outputs.json"

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)

    print("\nEvaluation finished successfully.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()