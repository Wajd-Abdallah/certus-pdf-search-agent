import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

BENCHMARK_FILE = Path(__file__).resolve().parent / "data" / "benchmark_subset.json"


@dataclass
class BenchmarkQuestion:
    question: str
    expected_answer: str | None = None
    expected_source: str | None = None
    expected_page: int | None = None
    should_abstain: bool = False


def load_small_benchmark() -> List[BenchmarkQuestion]:
    with BENCHMARK_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        BenchmarkQuestion(
            question=item["question"],
            expected_answer=item.get("expected_answer"),
            expected_source=item.get("expected_source"),
            expected_page=item.get("expected_page"),
            should_abstain=item.get("should_abstain", False),
        )
        for item in data
    ]