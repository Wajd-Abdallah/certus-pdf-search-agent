import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class BenchmarkQuestion:
    question: str
    expected_answer: str | None = None
    expected_source: str | None = None
    expected_page: int | None = None


def load_small_benchmark() -> List[BenchmarkQuestion]:
    benchmark_file = Path("evaluation/data/benchmark_subset.json")

    with benchmark_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        BenchmarkQuestion(
            question=item["question"],
            expected_answer=item.get("expected_answer"),
            expected_source=item.get("expected_source"),
            expected_page=item.get("expected_page"),
        )
        for item in data
    ]