import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

DATA_DIR = Path(__file__).resolve().parent / "data"

RETRIEVAL_BENCHMARK = DATA_DIR / "benchmark_retrieval_400.json"
FULL_BENCHMARK = DATA_DIR / "benchmark_full_120.json"

# Temporary compatibility for existing scripts.
BENCHMARK_FILE = FULL_BENCHMARK


@dataclass
class BenchmarkQuestion:
    question: str
    expected_answer: str | None = None
    expected_source: str | None = None
    expected_page: int | None = None
    should_abstain: bool = False


def _load(path: Path) -> List[BenchmarkQuestion]:
    with path.open("r", encoding="utf-8") as f:
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


def load_retrieval_benchmark() -> List[BenchmarkQuestion]:
    return _load(RETRIEVAL_BENCHMARK)


def load_full_benchmark() -> List[BenchmarkQuestion]:
    return _load(FULL_BENCHMARK)


# Backwards compatibility for old scripts.
def load_small_benchmark() -> List[BenchmarkQuestion]:
    return load_full_benchmark()