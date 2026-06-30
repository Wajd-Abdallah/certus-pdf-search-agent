import json
from pathlib import Path


def main():
    results_dir = Path("evaluation/results")

    summary_file = results_dir / "retrieval_summary.json"
    detailed_file = results_dir / "retrieval_outputs.json"

    with summary_file.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    with detailed_file.open("r", encoding="utf-8") as f:
        details = json.load(f)

    report_path = results_dir / "retrieval_report.md"

    with report_path.open("w", encoding="utf-8") as report:

        report.write("# Retrieval Evaluation Report\n\n")

        report.write("## Overall Metrics\n\n")
        report.write(f"- Number of questions: **{summary['num_questions']}**\n")
        report.write(f"- Recall@k: **{summary['Recall@k']:.3f}**\n")
        report.write(f"- MRR: **{summary['MRR']:.3f}**\n")
        report.write(f"- nDCG@k: **{summary['nDCG@k']:.3f}**\n\n")

        report.write("---\n\n")

        report.write("## Per Question Results\n\n")

        for i, item in enumerate(details, start=1):

            report.write(f"### Question {i}\n\n")

            report.write(f"**Question:** {item['question']}\n\n")

            report.write("**Metrics**\n\n")

            metrics = item["metrics"]

            report.write(f"- Recall@k: {metrics['Recall@k']:.3f}\n")
            report.write(f"- MRR: {metrics['MRR']:.3f}\n")
            report.write(f"- nDCG@k: {metrics['nDCG@k']:.3f}\n\n")

            report.write(
                f"Retrieved Chunks: **{len(item['retrieved_chunks'])}**\n\n"
            )

            report.write("---\n\n")

    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()