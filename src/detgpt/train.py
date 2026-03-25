from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def cxcywh_to_xyxy(box: list[float]) -> list[float]:
    """Convert a box from center format to corner format.

    Args:
        box: Bounding box as [center_x, center_y, width, height].

    Returns:
        Bounding box as [x1, y1, x2, y2].
    """
    cx, cy, w, h = box
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two boxes in cxcywh format.

    Args:
        box1: First bounding box as [center_x, center_y, width, height].
        box2: Second bounding box as [center_x, center_y, width, height].

    Returns:
        Intersection over union.
    """
    x1_a, y1_a, x2_a, y2_a = cxcywh_to_xyxy(box1)
    x1_b, y1_b, x2_b, y2_b = cxcywh_to_xyxy(box2)

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union


def compute_precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """Compute precision and recall.

    Args:
        tp: Number of true positives.
        fp: Number of false positives.
        fn: Number of false negatives.

    Returns:
        Tuple of (precision, recall).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def validate_record(record: dict[str, Any], require_scores: bool = False) -> None:
    """Validate a prediction or ground-truth record.

    Args:
        record: Input record.
        require_scores: Whether the record must include scores.

    Raises:
        ValueError: If the record has an invalid structure.
    """
    required_keys = {"boxes", "labels"}
    missing = required_keys - set(record)
    if missing:
        raise ValueError(f"Record is missing required keys: {sorted(missing)}")

    boxes = record["boxes"]
    labels = record["labels"]

    if not isinstance(boxes, list) or not isinstance(labels, list):
        raise ValueError("Record fields 'boxes' and 'labels' must be lists.")

    if len(boxes) != len(labels):
        raise ValueError("Record fields 'boxes' and 'labels' must have the same length.")

    for box in boxes:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Each box must be a list of four numbers in cxcywh format.")

    if require_scores:
        if "scores" not in record:
            raise ValueError("Prediction record must include 'scores'.")
        scores = record["scores"]
        if not isinstance(scores, list):
            raise ValueError("Prediction field 'scores' must be a list.")
        if len(scores) != len(boxes):
            raise ValueError("Prediction fields 'boxes' and 'scores' must have the same length.")


def build_gt_index(ground_truth: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index ground truth by image path.

    Args:
        ground_truth: Ground-truth records.

    Returns:
        Dictionary keyed by image_path.

    Raises:
        ValueError: If a ground-truth record has no image_path.
    """
    index: dict[str, dict[str, Any]] = {}
    for gt in ground_truth:
        validate_record(gt, require_scores=False)
        image_path = gt.get("image_path")
        if image_path is None:
            raise ValueError("Each ground-truth record must include 'image_path'.")
        index[str(image_path)] = gt
    return index


def evaluate_image(
    pred: dict[str, Any],
    gt: dict[str, Any],
    iou_threshold: float = 0.5,
) -> dict[str, int]:
    """Evaluate one image at a given IoU threshold.

    Args:
        pred: Prediction record with boxes, labels, and optionally scores.
        gt: Ground-truth record with boxes and labels.
        iou_threshold: IoU threshold for a match.

    Returns:
        Dictionary with tp, fp, and fn.
    """
    validate_record(pred, require_scores=False)
    validate_record(gt, require_scores=False)

    scores = pred.get("scores", [1.0] * len(pred["boxes"]))
    pred_items = list(zip(pred["boxes"], pred["labels"], scores, strict=False))
    pred_items.sort(key=lambda item: item[2], reverse=True)

    matched_gt_indices: set[int] = set()
    tp = 0
    fp = 0

    for pred_box, pred_label, _ in pred_items:
        found_match = False

        for gt_index, (gt_box, gt_label) in enumerate(zip(gt["boxes"], gt["labels"], strict=False)):
            if gt_index in matched_gt_indices:
                continue

            if pred_label != gt_label:
                continue

            iou = compute_iou(pred_box, gt_box)
            if iou >= iou_threshold:
                tp += 1
                matched_gt_indices.add(gt_index)
                found_match = True
                break

        if not found_match:
            fp += 1

    fn = len(gt["boxes"]) - len(matched_gt_indices)

    return {"tp": tp, "fp": fp, "fn": fn}


def evaluate_dataset_at_threshold(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, float | int]:
    """Evaluate a dataset at a given IoU threshold.

    Args:
        predictions: Prediction records.
        ground_truth: Ground-truth records.
        iou_threshold: IoU threshold for matching.

    Returns:
        Aggregated metrics dictionary.
    """
    gt_index = build_gt_index(ground_truth)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    seen_image_paths: set[str] = set()

    for pred in predictions:
        validate_record(pred, require_scores=False)

        image_path = pred.get("image_path")
        if image_path is None:
            raise ValueError("Each prediction record must include 'image_path'.")

        image_path = str(image_path)
        seen_image_paths.add(image_path)

        gt = gt_index.get(
            image_path,
            {
                "image_path": image_path,
                "boxes": [],
                "labels": [],
            },
        )

        result = evaluate_image(pred, gt, iou_threshold=iou_threshold)
        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

    for image_path, gt in gt_index.items():
        if image_path in seen_image_paths:
            continue
        total_fn += len(gt["boxes"])

    precision, recall = compute_precision_recall(total_tp, total_fp, total_fn)

    return {
        "iou_threshold": iou_threshold,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def evaluate_dataset(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Evaluate a dataset at AP50-like and AP75-like thresholds.

    Args:
        predictions: Prediction records.
        ground_truth: Ground-truth records.

    Returns:
        Dictionary containing AP50-like and AP75-like summaries.
    """
    ap50 = evaluate_dataset_at_threshold(predictions, ground_truth, iou_threshold=0.50)
    ap75 = evaluate_dataset_at_threshold(predictions, ground_truth, iou_threshold=0.75)

    return {
        "AP50_like": ap50,
        "AP75_like": ap75,
    }


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
