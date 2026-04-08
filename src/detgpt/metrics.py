from __future__ import annotations

import math
from typing import Any

from detgpt.box_utils import compute_iou_cxcywh


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


def _validate_finite_numeric(value: Any, field_name: str) -> float:
    """Validate that a value can be interpreted as a finite float."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, but received bool.")

    try:
        float_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be numeric.") from error

    if not math.isfinite(float_value):
        raise ValueError(f"{field_name} must be finite.")

    return float_value


def _validate_boxes_and_labels(record: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """Validate and return the required boxes/labels fields."""
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
        for coordinate in box:
            _validate_finite_numeric(coordinate, "Box coordinate")

    return boxes, labels


def _validate_scores(record: dict[str, Any], boxes: list[Any], require_scores: bool) -> None:
    """Validate prediction scores if required or present."""
    should_validate_scores = require_scores or "scores" in record
    if not should_validate_scores:
        return

    if "scores" not in record:
        raise ValueError("Prediction record must include 'scores'.")
    scores = record["scores"]
    if not isinstance(scores, list):
        raise ValueError("Prediction field 'scores' must be a list.")
    if len(scores) != len(boxes):
        raise ValueError("Prediction fields 'boxes' and 'scores' must have the same length.")

    for score in scores:
        _validate_finite_numeric(score, "Score value")


def validate_record(record: dict[str, Any], require_scores: bool = False) -> None:
    """Validate a prediction or ground-truth record.

    Args:
        record: Input record.
        require_scores: Whether the record must include scores.

    Raises:
        ValueError: If the record has an invalid structure.
    """
    boxes, _ = _validate_boxes_and_labels(record)
    _validate_scores(record, boxes, require_scores=require_scores)


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
    validate_record(pred, require_scores=("scores" in pred))
    validate_record(gt, require_scores=False)

    scores = pred.get("scores", [1.0] * len(pred["boxes"]))
    pred_items = list(zip(pred["boxes"], pred["labels"], scores, strict=True))
    pred_items.sort(key=lambda item: item[2], reverse=True)

    matched_gt_indices: set[int] = set()
    tp = 0
    fp = 0

    for pred_box, pred_label, _ in pred_items:
        found_match = False

        for gt_index, (gt_box, gt_label) in enumerate(zip(gt["boxes"], gt["labels"], strict=True)):
            if gt_index in matched_gt_indices:
                continue

            if pred_label != gt_label:
                continue

            iou = compute_iou_cxcywh(pred_box, gt_box)
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
        validate_record(pred, require_scores=("scores" in pred))

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
