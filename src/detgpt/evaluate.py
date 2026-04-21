from __future__ import annotations

import csv
import json
from datetime import datetime
from typing import Any

import torch
import typer
from loguru import logger
from torch.utils.data import DataLoader

from detgpt import OUTPUTS_DIR
from detgpt.data import Task1DetectionDataset, task1_collate_fn
from detgpt.metrics import evaluate_dataset
from detgpt.model import GroundingDINOHandler, QwenVLMHandler, YOLOWorldHandler

DINO_DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
QWEN_DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
YOLO_DEFAULT_MODEL_ID = "yolov8s-world.pt"


def _resolve_detector(
    detector_backend: str,
    model_id: str | None,
) -> tuple[str, str, GroundingDINOHandler | QwenVLMHandler | YOLOWorldHandler]:
    """Resolve backend name, model id, and detector instance."""
    normalized_backend = detector_backend.strip().lower()

    if normalized_backend == "grounding_dino":
        resolved_model_id = model_id or DINO_DEFAULT_MODEL_ID
        return normalized_backend, resolved_model_id, GroundingDINOHandler(model_id=resolved_model_id)

    if normalized_backend == "qwen_vlm":
        resolved_model_id = model_id or QWEN_DEFAULT_MODEL_ID
        return normalized_backend, resolved_model_id, QwenVLMHandler(model_id=resolved_model_id)

    if normalized_backend == "yolo_world":
        resolved_model_id = model_id or YOLO_DEFAULT_MODEL_ID
        return normalized_backend, resolved_model_id, YOLOWorldHandler(model_id=resolved_model_id)

    raise typer.BadParameter(
        f"Unsupported detector backend '{detector_backend}'. Use 'grounding_dino', 'qwen_vlm', or 'yolo_world'."
    )


def _extract_query_categories(category_names: list[str]) -> list[str]:
    """Build unique non-empty category list while preserving order."""
    seen_categories: set[str] = set()
    query_categories: list[str] = []

    for category_name in category_names:
        normalized_name = category_name.strip()
        if not normalized_name or normalized_name in seen_categories:
            continue
        seen_categories.add(normalized_name)
        query_categories.append(normalized_name)

    return query_categories


def _xyxy_to_cxcywh(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Convert Nx4 xyxy boxes to cxcywh."""
    if boxes_xyxy.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    return torch.stack([cx, cy, w, h], dim=1).to(torch.float32)


def _prediction_record(
    image_path: str,
    backend: str,
    detections: dict[str, Any],
) -> dict[str, Any]:
    """Build one prediction record for metrics.py."""
    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]

    if backend != "qwen_vlm":
        boxes = _xyxy_to_cxcywh(boxes)

    return {
        "image_path": image_path,
        "boxes": boxes.detach().cpu().tolist(),
        "labels": [str(label_name) for label_name in labels],
        "scores": scores.detach().cpu().tolist(),
    }


def _ground_truth_record(image_path: str, target: dict[str, Any]) -> dict[str, Any]:
    """Build one ground-truth record for metrics.py."""
    return {
        "image_path": image_path,
        "boxes": target["boxes"].detach().cpu().tolist(),
        "labels": [str(category_name) for category_name in target["category_names"]],
    }


def run_task1_baseline(
    split: str = typer.Option("val", help="Dataset split to evaluate: val or train."),
    limit: int = typer.Option(20, help="Number of samples to evaluate."),
    detector_backend: str = typer.Option(
        "yolo_world",
        help="Detector backend: grounding_dino, qwen_vlm, or yolo_world.",
    ),
    model_id: str | None = typer.Option(
        None,
        help="Model ID or checkpoint path. If omitted, a backend-specific default is used.",
    ),
) -> None:
    """Evaluate the selected detector backend on Task 1."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / "task1_results" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    normalized_split = split.strip().lower()
    if normalized_split not in {"val", "train"}:
        raise typer.BadParameter("Unsupported split. Use 'val' or 'train'.")

    dataset = Task1DetectionDataset(split=normalized_split, to_float=True)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=task1_collate_fn)

    normalized_backend, resolved_model_id, detector = _resolve_detector(
        detector_backend=detector_backend,
        model_id=model_id,
    )

    logger.info(
        "Running backend={} on split={} with model_id={}",
        normalized_backend,
        normalized_split,
        resolved_model_id,
    )

    predictions: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []
    summary_data: list[dict[str, Any]] = []

    for index, (images, targets) in enumerate(data_loader):
        if index >= limit:
            break

        image = images[0]
        target = targets[0]

        sample = dataset.samples[index]
        image_path = str(sample.get("local_path", ""))

        if not image_path:
            logger.warning("Skipping sample {} because local_path is missing.", index)
            continue

        query_categories = _extract_query_categories(target["category_names"])
        ground_truth.append(_ground_truth_record(image_path, target))

        if not query_categories:
            predictions.append(
                {
                    "image_path": image_path,
                    "boxes": [],
                    "labels": [],
                    "scores": [],
                }
            )
            summary_data.append(
                {
                    "image_id": target["image_id"].item() if "image_id" in target else index,
                    "num_gt": len(target["boxes"]),
                    "num_pred": 0,
                    "categories": "",
                    "avg_score": 0.0,
                }
            )
            continue

        if normalized_backend == "qwen_vlm":
            detections = detector.predict(
                image,
                query_categories,
                max_detections_per_category=1,
                temperature=0.0,
                return_debug_outputs=False,
            )
        else:
            detections = detector.predict(image, query_categories)

        predictions.append(_prediction_record(image_path, normalized_backend, detections))

        scores_tensor = detections["scores"]
        summary_data.append(
            {
                "image_id": target["image_id"].item() if "image_id" in target else index,
                "num_gt": len(target["boxes"]),
                "num_pred": len(detections["boxes"]),
                "categories": "|".join(query_categories),
                "avg_score": float(scores_tensor.mean().item()) if len(scores_tensor) > 0 else 0.0,
            }
        )

    metrics = evaluate_dataset(predictions=predictions, ground_truth=ground_truth)

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    summary_path = run_dir / "detections_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["image_id", "num_gt", "num_pred", "categories", "avg_score"],
        )
        writer.writeheader()
        writer.writerows(summary_data)

    logger.info("Saved metrics to {}", metrics_path)
    logger.info("Saved detection summary to {}", summary_path)
    logger.info(
        "AP50={:.4f}, AP75={:.4f}, mean_AP_50_75={:.4f}",
        float(metrics["AP50"]["ap"]),
        float(metrics["AP75"]["ap"]),
        float(metrics["mean_AP_50_75"]),
    )


if __name__ == "__main__":
    typer.run(run_task1_baseline)
