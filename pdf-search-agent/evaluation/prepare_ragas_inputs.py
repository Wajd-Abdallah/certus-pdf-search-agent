import json
from pathlib import Path


def prediction_to_ragas_row(prediction: dict, reference_answer: str | None = None) -> dict:
    return {
        "question": prediction.get("question", ""),
        "answer": prediction.get("answer", ""),
        "contexts": prediction.get("retrieved_contexts", []),
        "ground_truth": reference_answer or "",
        "abstained": prediction.get("abstained", False),
        "citations": prediction.get("citations", [])
    }


def predictions_to_ragas_rows(predictions: list, references: dict | None = None) -> list:
    rows = []

    for prediction in predictions:
        question = prediction.get("question", "")
        reference_answer = None

        if references:
            reference_answer = references.get(question)

        rows.append(prediction_to_ragas_row(prediction, reference_answer))

    return rows


def save_ragas_input(rows: list, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)


def load_predictions(path: str) -> list:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        return data.get("predictions", [])

    return data


def prepare_ragas_file(predictions_path: str, output_path: str, references_path: str | None = None) -> None:
    predictions = load_predictions(predictions_path)
    references = None

    if references_path:
        with open(references_path, "r", encoding="utf-8") as file:
            references = json.load(file)

    rows = predictions_to_ragas_rows(predictions, references)
    save_ragas_input(rows, output_path)