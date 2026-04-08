from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from detgpt.metrics import evaluate_dataset


def load_json(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON file containing a list of records.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed list of dictionaries.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of records.")

    return data


def save_json(data: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary to JSON.

    Args:
        data: Dictionary to save.
        path: Output path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def run_file_evaluation(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, dict[str, float | int]]:
    """Run evaluation from JSON files.

    Args:
        predictions_path: Path to predictions JSON.
        ground_truth_path: Path to ground-truth JSON.
        output_path: Optional output JSON path for results.

    Returns:
        Evaluation results.
    """
    predictions = load_json(predictions_path)
    ground_truth = load_json(ground_truth_path)

    results = evaluate_dataset(predictions, ground_truth)

    if output_path is not None:
        save_json(results, output_path)

    return results


if __name__ == "__main__":
    mock_predictions = [
        {
            "image_path": "data/test.jpg",
            "boxes": [
                [142.76349258422852, 91.74662780761719, 192.16155242919922, 95.01123046875],
            ],
            "scores": [0.8027151226997375],
            "labels": ["car"],
        }
    ]

    mock_ground_truth = [
        {
            "image_path": "data/test.jpg",
            "boxes": [
                [140.0, 90.0, 190.0, 100.0],
            ],
            "labels": ["car"],
        }
    ]

    results = evaluate_dataset(mock_predictions, mock_ground_truth)
    print(results)
