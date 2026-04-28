from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import typer
from loguru import logger

from detgpt import OUTPUTS_DIR
from detgpt.data import Task1DetectionDataset
from detgpt.metrics import evaluate_dataset
from detgpt.model import GroundingDINOHandler, QwenVLMHandler, YOLOWorldHandler

DINO_DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
QWEN_DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
YOLO_DEFAULT_MODEL_ID = "yolov8s-world.pt"


def _resolve_detector(detector_backend: str, model_id: str | None):
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


def _cxcywh_to_xyxy(boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    if boxes_cxcywh.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32)

    cx = boxes_cxcywh[:, 0]
    cy = boxes_cxcywh[:, 1]
    w = boxes_cxcywh[:, 2]
    h = boxes_cxcywh[:, 3]

    x1 = cx - (w / 2.0)
    y1 = cy - (h / 2.0)
    x2 = cx + (w / 2.0)
    y2 = cy + (h / 2.0)

    return torch.stack([x1, y1, x2, y2], dim=1).to(torch.float32)


def _prediction_record(image_path: str, backend: str, detections: dict[str, Any]) -> dict[str, Any]:
    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]

    # YOLO and Grounding DINO return xyxy; Qwen returns cxcywh.
    if backend != "qwen_vlm":
        boxes = _xyxy_to_cxcywh(boxes)

    return {
        "image_path": image_path,
        "boxes": boxes.detach().cpu().tolist(),
        "labels": [str(label_name) for label_name in labels],
        "scores": scores.detach().cpu().tolist(),
    }


def _ground_truth_record(image_path: str, target: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_path": image_path,
        "boxes": target["boxes"].detach().cpu().tolist(),
        "labels": [str(category_name) for category_name in target["category_names"]],
    }


def _empty_detections() -> dict[str, Any]:
    return {
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "scores": torch.zeros((0,), dtype=torch.float32),
        "labels": [],
    }


def _predict_detections(
    detector,
    normalized_backend: str,
    image: torch.Tensor,
    query_categories: list[str],
    qwen_max_detections_per_category: int,
    qwen_max_new_tokens: int,
    qwen_temperature: float,
    qwen_debug_dump: bool,
) -> dict[str, Any]:
    if not query_categories:
        return _empty_detections()

    if normalized_backend == "qwen_vlm":
        return detector.predict(
            image,
            query_categories,
            max_detections_per_category=qwen_max_detections_per_category,
            max_new_tokens=qwen_max_new_tokens,
            temperature=qwen_temperature,
            return_debug_outputs=qwen_debug_dump,
        )

    return detector.predict(image, query_categories)


def _save_visualization(
    image_tensor: torch.Tensor,
    detections: dict[str, Any],
    backend: str,
    output_path: Path,
) -> None:
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()

    boxes = detections["boxes"].detach().cpu().to(torch.float32)
    scores = detections["scores"].detach().cpu().to(torch.float32)
    labels = [str(label) for label in detections["labels"]]

    boxes_xyxy = _cxcywh_to_xyxy(boxes) if backend == "qwen_vlm" else boxes

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)

    for box, score, label in zip(boxes_xyxy.tolist(), scores.tolist(), labels, strict=True):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(y1 - 2, 0),
            f"{label}: {score:.2f}",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            verticalalignment="bottom",
        )

    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _record_summary(
    summary_data: list[dict[str, Any]],
    target: dict[str, Any],
    detections: dict[str, Any],
    query_categories: list[str],
    index: int,
) -> None:
    scores_tensor = detections["scores"]
    summary_data.append(
        {
            "dataset_index": index,
            "image_id": target["image_id"].item() if "image_id" in target else index,
            "num_gt": len(target["boxes"]),
            "num_pred": len(detections["boxes"]),
            "categories": "|".join(query_categories),
            "avg_score": float(scores_tensor.mean().item()) if len(scores_tensor) > 0 else 0.0,
        }
    )


def _save_results(run_dir: Path, metrics: dict[str, Any], summary_data: list[dict[str, Any]]) -> None:
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    summary_path = run_dir / "detections_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=["dataset_index", "image_id", "num_gt", "num_pred", "categories", "avg_score"],
        )
        writer.writeheader()
        writer.writerows(summary_data)

    logger.info("Saved metrics to {}", metrics_path)
    logger.info("Saved detection summary to {}", summary_path)


def _save_qwen_debug_dump(
    run_dir: Path,
    normalized_backend: str,
    qwen_debug_dump: bool,
    qwen_debug_entries: list[dict[str, Any]],
) -> None:
    if not qwen_debug_dump:
        return

    if normalized_backend != "qwen_vlm":
        logger.warning("Ignoring --qwen-debug-dump because detector_backend != 'qwen_vlm'.")
        return

    debug_path = run_dir / "qwen_debug_dump.json"
    with debug_path.open("w", encoding="utf-8") as file_handle:
        json.dump(qwen_debug_entries, file_handle, indent=2)

    logger.info("Saved Qwen debug dump to {}", debug_path)


def _sample_balanced_indices(
    dataset: Task1DetectionDataset,
    samples_per_class: int,
    seed: int,
    limit: int,
) -> list[int]:
    rng = random.Random(seed)

    per_class: dict[str, list[int]] = defaultdict(list)

    for index, sample in enumerate(dataset.samples):
        classes = {
            str(annotation.get("category_name", "")).strip()
            for annotation in sample.get("annotations", [])
            if str(annotation.get("category_name", "")).strip()
        }

        for class_name in classes:
            per_class[class_name].append(index)

    selected_indices: list[int] = []
    used_indices: set[int] = set()

    class_names = sorted(per_class)
    logger.info("Balanced sampling over {} classes: {}", len(class_names), class_names)

    for class_name in class_names:
        candidate_indices = per_class[class_name][:]
        rng.shuffle(candidate_indices)

        added_for_class = 0
        for index in candidate_indices:
            if index in used_indices:
                continue

            selected_indices.append(index)
            used_indices.add(index)
            added_for_class += 1

            if added_for_class >= samples_per_class:
                break

        logger.info(
            "Selected {} unique images for class '{}' from {} candidates.",
            added_for_class,
            class_name,
            len(candidate_indices),
        )

    rng.shuffle(selected_indices)

    if limit > 0:
        selected_indices = selected_indices[:limit]

    logger.info("Balanced selected image count: {}", len(selected_indices))
    return selected_indices


def _sequential_indices(dataset: Task1DetectionDataset, limit: int) -> list[int]:
    max_count = len(dataset) if limit <= 0 else min(limit, len(dataset))
    return list(range(max_count))


def _process_single_sample(
    *,
    index: int,
    image: torch.Tensor,
    target: dict[str, Any],
    dataset: Task1DetectionDataset,
    detector,
    normalized_backend: str,
    qwen_max_detections_per_category: int,
    qwen_max_new_tokens: int,
    qwen_temperature: float,
    qwen_debug_dump: bool,
    save_viz: bool,
    viz_dir: Path | None,
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    summary_data: list[dict[str, Any]],
    qwen_debug_entries: list[dict[str, Any]],
) -> None:
    sample = dataset.samples[index]
    image_path = str(sample.get("local_path", ""))

    if not image_path:
        logger.warning("Skipping sample {} because local_path is missing.", index)
        return

    query_categories = _extract_query_categories(target["category_names"])

    detections = _predict_detections(
        detector=detector,
        normalized_backend=normalized_backend,
        image=image,
        query_categories=query_categories,
        qwen_max_detections_per_category=qwen_max_detections_per_category,
        qwen_max_new_tokens=qwen_max_new_tokens,
        qwen_temperature=qwen_temperature,
        qwen_debug_dump=qwen_debug_dump,
    )

    ground_truth.append(_ground_truth_record(image_path, target))
    predictions.append(_prediction_record(image_path, normalized_backend, detections))

    if qwen_debug_dump and normalized_backend == "qwen_vlm" and "debug_entries" in detections:
        qwen_debug_entries.append(
            {
                "image_path": image_path,
                "image_id": target["image_id"].item() if "image_id" in target else index,
                "categories": query_categories,
                "debug_entries": detections["debug_entries"],
            }
        )

    if save_viz and viz_dir is not None:
        image_id = target["image_id"].item() if "image_id" in target else index
        viz_path = viz_dir / f"{index:04d}_image_{image_id}.png"
        _save_visualization(
            image_tensor=image,
            detections=detections,
            backend=normalized_backend,
            output_path=viz_path,
        )

    _record_summary(
        summary_data=summary_data,
        target=target,
        detections=detections,
        query_categories=query_categories,
        index=index,
    )


def run_task1_baseline(
    split: str = typer.Option("val", help="Dataset split to evaluate: val or train."),
    limit: int = typer.Option(20, help="Number of samples to evaluate. Use 0 for all selected samples."),
    detector_backend: str = typer.Option(
        "yolo_world",
        help="Detector backend: grounding_dino, qwen_vlm, or yolo_world.",
    ),
    model_id: str | None = typer.Option(
        None,
        help="Model ID or checkpoint path. If omitted, a backend-specific default is used.",
    ),
    save_results: bool = typer.Option(
        True,
        "--save-results/--no-save-results",
        help="Save metrics.json and detections_summary.csv.",
    ),
    save_viz: bool = typer.Option(
        False,
        "--save-viz/--no-save-viz",
        help="Save detection visualizations for each evaluated sample.",
    ),
    balanced: bool = typer.Option(
        False,
        "--balanced/--no-balanced",
        help="Use class-balanced image sampling instead of the first N images.",
    ),
    samples_per_class: int = typer.Option(
        10,
        help="Images to sample per class when --balanced is enabled.",
    ),
    seed: int = typer.Option(
        42,
        help="Random seed for balanced sampling.",
    ),
    qwen_max_detections_per_category: int = typer.Option(
        1,
        help="Maximum detections per category for Qwen-VLM.",
    ),
    qwen_max_new_tokens: int = typer.Option(
        256,
        help="Maximum generated tokens for Qwen-VLM.",
    ),
    qwen_temperature: float = typer.Option(
        0.0,
        help="Generation temperature for Qwen-VLM.",
    ),
    qwen_debug_dump: bool = typer.Option(
        False,
        "--qwen-debug-dump/--no-qwen-debug-dump",
        help="Save raw Qwen prompts, generations, and parser outputs to JSON.",
    ),
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / "task1_results" / f"run_{timestamp}"

    should_create_run_dir = save_results or save_viz or qwen_debug_dump
    if should_create_run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

    normalized_split = split.strip().lower()
    if normalized_split not in {"val", "train"}:
        raise typer.BadParameter("Unsupported split. Use 'val' or 'train'.")

    if samples_per_class < 1:
        raise typer.BadParameter("--samples-per-class must be at least 1.")

    dataset = Task1DetectionDataset(split=normalized_split, to_float=True)

    normalized_backend, resolved_model_id, detector = _resolve_detector(
        detector_backend=detector_backend,
        model_id=model_id,
    )

    if balanced:
        selected_indices = _sample_balanced_indices(
            dataset=dataset,
            samples_per_class=samples_per_class,
            seed=seed,
            limit=limit,
        )
    else:
        selected_indices = _sequential_indices(dataset=dataset, limit=limit)

    logger.info(
        "Running backend={} on split={} with model_id={} balanced={} selected_images={}",
        normalized_backend,
        normalized_split,
        resolved_model_id,
        balanced,
        len(selected_indices),
    )

    predictions: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []
    summary_data: list[dict[str, Any]] = []
    qwen_debug_entries: list[dict[str, Any]] = []

    viz_dir = run_dir / "visualizations" if save_viz else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    for index in selected_indices:
        image, target = dataset[index]

        _process_single_sample(
            index=index,
            image=image,
            target=target,
            dataset=dataset,
            detector=detector,
            normalized_backend=normalized_backend,
            qwen_max_detections_per_category=qwen_max_detections_per_category,
            qwen_max_new_tokens=qwen_max_new_tokens,
            qwen_temperature=qwen_temperature,
            qwen_debug_dump=qwen_debug_dump,
            save_viz=save_viz,
            viz_dir=viz_dir,
            predictions=predictions,
            ground_truth=ground_truth,
            summary_data=summary_data,
            qwen_debug_entries=qwen_debug_entries,
        )

    metrics = evaluate_dataset(predictions=predictions, ground_truth=ground_truth)

    if save_results:
        _save_results(run_dir=run_dir, metrics=metrics, summary_data=summary_data)

    if save_viz and viz_dir is not None:
        logger.info("Saved visualizations to {}", viz_dir)

    _save_qwen_debug_dump(
        run_dir=run_dir,
        normalized_backend=normalized_backend,
        qwen_debug_dump=qwen_debug_dump,
        qwen_debug_entries=qwen_debug_entries,
    )

    logger.info(
        "AP50={:.4f}, AP75={:.4f}, mean_AP_50_75={:.4f}",
        float(metrics["AP50"]["ap"]),
        float(metrics["AP75"]["ap"]),
        float(metrics["mean_AP_50_75"]),
    )


if __name__ == "__main__":
    typer.run(run_task1_baseline)
