"""
Robustness smoke tests: checks that the system handles edge cases and
invalid input gracefully (no crashes, clear errors) instead of raising
unhandled exceptions.

Run from the project root:
    python3 -m evaluation.robustness_eval
"""

import json
from pathlib import Path

from app.pipeline import process_pdf, answer_question


def run_case(name, fn):
    print("=" * 60)
    print("Case:", name)
    try:
        result = fn()
        print("Result:", result)
        return {"case": name, "passed": True, "result": result}
    except Exception as e:
        print("UNHANDLED EXCEPTION:", e)
        return {"case": name, "passed": False, "error": str(e)}


def main():
    tmp_dir = Path("evaluation/results/robustness_files")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    empty_pdf = tmp_dir / "empty.pdf"
    empty_pdf.write_bytes(b"")

    fake_pdf = tmp_dir / "fake.pdf"
    fake_pdf.write_text("This is just a text file renamed to .pdf, not a real PDF.")

    garbage_pdf = tmp_dir / "garbage.pdf"
    garbage_pdf.write_bytes(bytes([0, 1, 2, 3, 4, 5]) * 100)

    results = []

    results.append(run_case(
        "Upload: empty file",
        lambda: process_pdf(str(empty_pdf)),
    ))
    results.append(run_case(
        "Upload: text file renamed to .pdf",
        lambda: process_pdf(str(fake_pdf)),
    ))
    results.append(run_case(
        "Upload: random binary garbage named .pdf",
        lambda: process_pdf(str(garbage_pdf)),
    ))
    results.append(run_case(
        "Upload: nonexistent file path",
        lambda: process_pdf(str(tmp_dir / "does_not_exist.pdf")),
    ))
    results.append(run_case(
        "Question: empty string",
        lambda: answer_question(""),
    ))
    results.append(run_case(
        "Question: whitespace only",
        lambda: answer_question("     "),
    ))
    results.append(run_case(
        "Question: extremely long nonsense",
        lambda: answer_question(" ".join(["banana"] * 2000)),
    ))
    results.append(run_case(
        "Question: different language (German)",
        lambda: answer_question("Was ist der Hauptbeitrag dieses Papiers zur Merkmalsauswahl?"),
    ))

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    print("\n" + "=" * 60)
    print("Robustness Summary")
    print("=" * 60)
    print(f"Passed (handled gracefully, no crash): {passed}/{total}")
    for r in results:
        status = "OK" if r["passed"] else "CRASH"
        print(f"  [{status}] {r['case']}")

    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "robustness_outputs.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nSaved to: {output_dir}/robustness_outputs.json")


if __name__ == "__main__":
    main()