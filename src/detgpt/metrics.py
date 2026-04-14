from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from detgpt.box_utils import compute_iou_cxcywh


def compute_precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """Compute precision and recall from TP/FP/FN counts."""
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
    """Validate and return boxes and labels."""
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
    """Validate a prediction or ground-truth record."""
    boxes, _ = _validate_boxes_and_labels(record)
    _validate_scores(record, boxes, require_scores=require_scores)

    image_path = record.get("image_path")
    if image_path is None:
        raise ValueError("Each record must include 'image_path'.")


def build_gt_index(ground_truth: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index ground truth by image_path."""
    index: dict[str, dict[str, Any]] = {}

    for gt in ground_truth:
        validate_record(gt, require_scores=False)
        image_path = str(gt["image_path"])

        if image_path in index:
            raise ValueError(f"Duplicate ground-truth image_path found: {image_path}")

        index[image_path] = gt

    return index


def _compute_ap_from_pr(precisions: list[float], recalls: list[float]) -> float:
    """Compute AP from a precision-recall curve using precision-envelope integration."""
    if not precisions or not recalls:
        return 0.0

    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]

    for index in range(len(mpre) - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])

    ap = 0.0
    for index in range(1, len(mrec)):
        delta_recall = mrec[index] - mrec[index - 1]
        if delta_recall > 0:
            ap += delta_recall * mpre[index]

    return ap


def _extract_all_classes(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> list[str]:
    """Extract sorted unique class labels from predictions and ground truth."""
    classes: set[str] = set()

    for gt in ground_truth:
        validate_record(gt, require_scores=False)
        for label in gt["labels"]:
            classes.add(str(label))

    for pred in predictions:
        validate_record(pred, require_scores=True)
        for label in pred["labels"]:
            classes.add(str(label))

    return sorted(classes)


def _count_gt_for_class(
    ground_truth: list[dict[str, Any]],
    class_name: str,
) -> int:
    """Count ground-truth boxes for a class across the dataset."""
    total = 0
    for gt in ground_truth:
        for label in gt["labels"]:
            if str(label) == class_name:
                total += 1
    return total


def _prepare_predictions_for_class(
    predictions: list[dict[str, Any]],
    class_name: str,
) -> list[dict[str, Any]]:
    """Flatten predictions for one class into a globally score-sorted list."""
    entries: list[dict[str, Any]] = []

    for pred in predictions:
        validate_record(pred, require_scores=True)
        image_path = str(pred["image_path"])

        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"], strict=True):
            if str(label) != class_name:
                continue

            entries.append(
                {
                    "image_path": image_path,
                    "box": box,
                    "score": float(score),
                }
            )

    entries.sort(key=lambda item: item["score"], reverse=True)
    return entries


def evaluate_class_at_threshold(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    class_name: str,
    iou_threshold: float,
) -> dict[str, float | int | str]:
    """Evaluate one class at one IoU threshold."""
    gt_index_by_image = build_gt_index(ground_truth)
    total_gt = _count_gt_for_class(ground_truth, class_name)
    pred_entries = _prepare_predictions_for_class(predictions, class_name)

    matched_gt_indices_by_image: dict[str, set[int]] = defaultdict(set)

    tp_flags: list[int] = []
    fp_flags: list[int] = []

    for pred_entry in pred_entries:
        image_path = pred_entry["image_path"]
        pred_box = pred_entry["box"]

        gt_record = gt_index_by_image.get(
            image_path,
            {
                "image_path": image_path,
                "boxes": [],
                "labels": [],
            },
        )

        gt_boxes = gt_record["boxes"]
        gt_labels = gt_record["labels"]

        best_iou = -1.0
        best_gt_match_index = -1
        used_gt_indices = matched_gt_indices_by_image[image_path]

        for gt_match_index, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels, strict=True)):
            if gt_match_index in used_gt_indices:
                continue
            if str(gt_label) != class_name:
                continue

            iou = compute_iou_cxcywh(pred_box, gt_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_match_index = gt_match_index

        if best_gt_match_index >= 0:
            used_gt_indices.add(best_gt_match_index)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)

    cumulative_tp: list[int] = []
    cumulative_fp: list[int] = []

    running_tp = 0
    running_fp = 0
    for tp_flag, fp_flag in zip(tp_flags, fp_flags, strict=True):
        running_tp += tp_flag
        running_fp += fp_flag
        cumulative_tp.append(running_tp)
        cumulative_fp.append(running_fp)

    precisions_curve: list[float] = []
    recalls_curve: list[float] = []

    for cumulative_tp_value, cumulative_fp_value in zip(cumulative_tp, cumulative_fp, strict=True):
        precision = (
            cumulative_tp_value / (cumulative_tp_value + cumulative_fp_value)
            if (cumulative_tp_value + cumulative_fp_value) > 0
            else 0.0
        )
        recall = cumulative_tp_value / total_gt if total_gt > 0 else 0.0
        precisions_curve.append(precision)
        recalls_curve.append(recall)

    ap = _compute_ap_from_pr(precisions_curve, recalls_curve) if total_gt > 0 else 0.0

    tp = running_tp
    fp = running_fp
    fn = total_gt - tp
    precision, recall = compute_precision_recall(tp, fp, fn)

    return {
        "class_name": class_name,
        "iou_threshold": iou_threshold,
        "ap": ap,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_predictions": len(pred_entries),
        "num_ground_truth": total_gt,
    }


def evaluate_dataset_at_threshold(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float,
) -> dict[str, Any]:
    """Evaluate dataset at a given IoU threshold using class-wise AP and macro averaging."""
    classes = _extract_all_classes(predictions, ground_truth)

    per_class: list[dict[str, float | int | str]] = []
    valid_class_aps: list[float] = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0

    for class_name in classes:
        class_result = evaluate_class_at_threshold(
            predictions=predictions,
            ground_truth=ground_truth,
            class_name=class_name,
            iou_threshold=iou_threshold,
        )
        per_class.append(class_result)

        class_gt = int(class_result["num_ground_truth"])
        if class_gt > 0:
            valid_class_aps.append(float(class_result["ap"]))

        total_tp += int(class_result["tp"])
        total_fp += int(class_result["fp"])
        total_fn += int(class_result["fn"])
        total_gt += int(class_result["num_ground_truth"])
        total_pred += int(class_result["num_predictions"])

    macro_ap = sum(valid_class_aps) / len(valid_class_aps) if valid_class_aps else 0.0
    micro_precision, micro_recall = compute_precision_recall(total_tp, total_fp, total_fn)

    return {
        "iou_threshold": iou_threshold,
        "ap": macro_ap,
        "precision": micro_precision,
        "recall": micro_recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "num_predictions": total_pred,
        "num_ground_truth": total_gt,
        "num_classes": len(classes),
        "num_classes_with_gt": len(valid_class_aps),
        "per_class": per_class,
    }


def evaluate_image(
    pred: dict[str, Any],
    gt: dict[str, Any],
    iou_threshold: float = 0.5,
) -> dict[str, int]:
    """Evaluate one image at a given IoU threshold for debugging only."""
    validate_record(pred, require_scores=("scores" in pred))
    validate_record(gt, require_scores=False)

    scores = pred.get("scores", [1.0] * len(pred["boxes"]))
    pred_items = list(zip(pred["boxes"], pred["labels"], scores, strict=True))
    pred_items.sort(key=lambda item: float(item[2]), reverse=True)

    matched_gt_indices: set[int] = set()
    tp = 0
    fp = 0

    for pred_box, pred_label, _ in pred_items:
        best_iou = -1.0
        best_gt_match_index = -1

        for gt_match_index, (gt_box, gt_label) in enumerate(zip(gt["boxes"], gt["labels"], strict=True)):
            if gt_match_index in matched_gt_indices:
                continue
            if str(pred_label) != str(gt_label):
                continue

            iou = compute_iou_cxcywh(pred_box, gt_box)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_match_index = gt_match_index

        if best_gt_match_index >= 0:
            matched_gt_indices.add(best_gt_match_index)
            tp += 1
        else:
            fp += 1

    fn = len(gt["boxes"]) - len(matched_gt_indices)
    return {"tp": tp, "fp": fp, "fn": fn}


def evaluate_dataset(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate dataset at AP50 and AP75 with class-wise AP averaging."""
    ap50 = evaluate_dataset_at_threshold(predictions, ground_truth, iou_threshold=0.50)
    ap75 = evaluate_dataset_at_threshold(predictions, ground_truth, iou_threshold=0.75)

    return {
        "AP50": ap50,
        "AP75": ap75,
        "mean_AP_50_75": (float(ap50["ap"]) + float(ap75["ap"])) / 2.0,
    }
