BENCHMARK_FILE = Path(__file__).resolve().parent / "data" / "benchmark_subset.json"

def load_small_benchmark() -> List[BenchmarkQuestion]:
    with BENCHMARK_FILE.open("r", encoding="utf-8") as f:
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