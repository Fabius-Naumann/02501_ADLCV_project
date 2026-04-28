from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import typer
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader

from detgpt import OUTPUTS_DIR
from detgpt.data import Task1DetectionDataset, task1_collate_fn
from detgpt.metrics import evaluate_dataset
from detgpt.model import GroundingDINOHandler, QwenVLMHandler, YOLOWorldHandler
from detgpt.support_samples import (
    cropped_side_by_side,
    find_support_indices,
    marked_side_by_side,
    side_by_side,
)

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


def _cxcywh_to_xyxy(boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    """Convert Nx4 cxcywh boxes to xyxy."""
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


def _prediction_record(
    image_path: str,
    backend: str,
    detections: dict[str, Any],
) -> dict[str, Any]:
    """Build one prediction record for metrics.py in cxcywh format."""
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


def _empty_detections() -> dict[str, Any]:
    """Return an empty detection dictionary."""
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
    """Run backend-specific prediction."""
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
    """Save one visualization image with predicted boxes."""
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()

    boxes = detections["boxes"].detach().cpu().to(torch.float32)
    scores = detections["scores"].detach().cpu().to(torch.float32)
    labels = [str(label) for label in detections["labels"]]

    boxes_xyxy = _cxcywh_to_xyxy(boxes) if backend == "qwen_vlm" else boxes

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)

    for box, score, label in zip(boxes_xyxy.tolist(), scores.tolist(), labels, strict=True):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
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
    """Append one summary row."""
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


def _save_results(
    run_dir: Path,
    metrics: dict[str, Any],
    summary_data: list[dict[str, Any]],
) -> None:
    """Save metrics and summary CSV."""
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


def _save_qwen_debug_dump(
    run_dir: Path,
    normalized_backend: str,
    qwen_debug_dump: bool,
    qwen_debug_entries: list[dict[str, Any]],
) -> None:
    """Save Qwen debug JSON when requested."""
    if not qwen_debug_dump:
        return

    if normalized_backend != "qwen_vlm":
        logger.warning("Ignoring --qwen-debug-dump because detector_backend != 'qwen_vlm'.")
        return

    debug_path = run_dir / "qwen_debug_dump.json"
    with debug_path.open("w", encoding="utf-8") as file_handle:
        json.dump(qwen_debug_entries, file_handle, indent=2)
    logger.info("Saved Qwen debug dump to {}", debug_path)


def _ground_truth_record_for_category(
    image_path: str,
    target: dict[str, Any],
    category_name: str,
) -> dict[str, Any]:
    """Build one category-filtered ground-truth record in cxcywh format."""
    category_boxes = [
        target["boxes"][index].detach().cpu().tolist()
        for index, label in enumerate(target["category_names"])
        if str(label).casefold() == category_name.casefold()
    ]
    return {
        "image_path": image_path,
        "boxes": category_boxes,
        "labels": [category_name] * len(category_boxes),
    }


def _save_task2_results(
    run_dir: Path,
    metrics_by_method: dict[str, Any],
    method_rows: list[dict[str, Any]],
) -> None:
    """Persist Task 2 metrics and summary CSV."""
    metrics_path = run_dir / "task2_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics_by_method, file_handle, indent=2)

    summary_path = run_dir / "task2_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "method",
                "num_eval_pairs",
                "ap50",
                "ap75",
                "mean_ap_50_75",
            ],
        )
        writer.writeheader()
        writer.writerows(method_rows)

    logger.info("Saved Task 2 metrics to {}", metrics_path)
    logger.info("Saved Task 2 summary to {}", summary_path)


def _process_single_sample(
    *,
    index: int,
    images: list[torch.Tensor],
    targets: list[dict[str, Any]],
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
    """Process one dataset sample."""
    image = images[0]
    target = targets[0]

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


def run_task2_support_strategy_baseline(  # noqa: C901
    query_split: str = typer.Option("val", help="Query split to evaluate: val or train."),
    support_split: str = typer.Option("train", help="Support split for 1-shot/5-shot examples: val or train."),
    limit: int = typer.Option(20, help="Number of query images to evaluate."),
    category_names: str = typer.Option(
        ...,
        help="Comma-separated category names to evaluate (required).",
    ),
    qwen_model_id: str = typer.Option(
        QWEN_DEFAULT_MODEL_ID,
        help="Qwen model id for support-conditioned VLM detection.",
    ),
    qwen_max_detections_per_category: int = typer.Option(
        1,
        help="Maximum detections per category for Qwen methods.",
    ),
    localization_max_new_tokens: int = typer.Option(
        256,
        help="Maximum generated tokens for Qwen localization generation.",
    ),
    qwen_temperature: float = typer.Option(
        0.0,
        help="Generation temperature for Qwen methods.",
    ),
    save_results: bool = typer.Option(
        True,
        "--save-results/--no-save-results",
        help="Save Task 2 metrics and CSV summary.",
    ),
) -> dict[str, Any]:
    """Evaluate support-presentation strategies for selected classes."""
    normalized_query_split = query_split.strip().lower()
    normalized_support_split = support_split.strip().lower()
    if normalized_query_split not in {"val", "train"}:
        raise typer.BadParameter("Unsupported query split. Use 'val' or 'train'.")
    if normalized_support_split not in {"val", "train"}:
        raise typer.BadParameter("Unsupported support split. Use 'val' or 'train'.")
    if limit < 1:
        raise typer.BadParameter("--limit must be at least 1.")

    requested_categories = _extract_query_categories(category_names.split(","))
    if not requested_categories:
        raise typer.BadParameter("--category-names must include at least one non-empty category name.")
    requested_category_set = {name.casefold() for name in requested_categories}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / "task2_results" / f"run_{timestamp}"
    viz_dir = run_dir / "visualizations"
    if save_results:
        run_dir.mkdir(parents=True, exist_ok=True)
        viz_dir.mkdir(parents=True, exist_ok=True)

    query_dataset = Task1DetectionDataset(split=normalized_query_split, to_float=True)
    support_dataset = Task1DetectionDataset(split=normalized_support_split, to_float=True)
    query_loader = DataLoader(query_dataset, batch_size=1, collate_fn=task1_collate_fn)

    qwen_handler = QwenVLMHandler(model_id=qwen_model_id)
    qwen_task2_detection_system_prompts = {
        "side_by_side": qwen_handler._TASK2_OBJECT_DETECTION_BOUNDED_BOXES,
        "cropped_exemplars": qwen_handler._TASK2_OBJECT_DETECTION_CROPPED,
        "set_of_mark_visual": qwen_handler._TASK2_OBJECT_DETECTION_MARKED,
    }

    # Map strategy names to their panel builder functions (target_img=None for support panel only)
    strategy_builders = {
        "side_by_side": side_by_side,
        "cropped_exemplars": cropped_side_by_side,
        "set_of_mark_visual": marked_side_by_side,
    }
    shots = [1, 3, 5]
    max_requested_support = max(shots)

    method_predictions: dict[str, list[dict[str, Any]]] = {}
    method_ground_truth: dict[str, list[dict[str, Any]]] = {}

    def _append_eval(method_name: str, gt_record: dict[str, Any], pred_record: dict[str, Any]) -> None:
        method_predictions.setdefault(method_name, []).append(pred_record)
        method_ground_truth.setdefault(method_name, []).append(gt_record)

    total_query_images = 0
    total_category_pairs = 0

    for query_index, (images, targets) in enumerate(query_loader):
        if query_index >= limit:
            break
        total_query_images += 1

        query_image = images[0]
        query_target = targets[0]
        image_path = str(query_dataset.samples[query_index].get("local_path", ""))
        if not image_path:
            logger.warning("Skipping query sample {} due to missing local_path.", query_index)
            continue

        query_categories = [
            category_name
            for category_name in _extract_query_categories(query_target["category_names"])
            if category_name.casefold() in requested_category_set
        ]

        for category_name in query_categories:
            image_key = f"{image_path}::query={query_index}::category={category_name}"
            gt_record = _ground_truth_record_for_category(
                image_path=image_key,
                target=query_target,
                category_name=category_name,
            )
            if len(gt_record["boxes"]) == 0:
                continue

            total_category_pairs += 1

            support_query_index = query_index if normalized_support_split == normalized_query_split else -1
            support_indices = find_support_indices(
                dataset=support_dataset,
                category_name=category_name,
                query_index=support_query_index,
                n_support=max_requested_support,
            )

            query_height = int(query_image.shape[1])
            query_width = int(query_image.shape[2])

            for shot in shots:
                if len(support_indices) < shot:
                    continue

                support_samples = [support_dataset[support_index] for support_index in support_indices[:shot]]
                for strategy_name, strategy_builder in strategy_builders.items():
                    system_prompt = qwen_task2_detection_system_prompts[strategy_name]

                    # Build support panel without query image (target_img=None)
                    support_panel = strategy_builder(
                        target_img=None,
                        n_support_img=support_samples,
                        support_category_name=category_name,
                    )

                    if not isinstance(support_panel, Image.Image):
                        raise TypeError(f"Support strategy '{strategy_name}' did not return a PIL.Image instance.")

                    # Run prediction with support panel and query image
                    support_detections = qwen_handler.predict_with_support_panel(
                        query_image_tensor=query_image,
                        support_panel_pil=support_panel,
                        category_name=category_name,
                        query_image_width=query_width,
                        query_image_height=query_height,
                        max_detections=qwen_max_detections_per_category,
                        max_new_tokens=localization_max_new_tokens,
                        temperature=qwen_temperature,
                        return_debug_outputs=False,
                        system_prompt=system_prompt,
                    )

                    _append_eval(
                        method_name=f"vlm_support_{strategy_name}_{shot}shot",
                        gt_record=gt_record,
                        pred_record=_prediction_record(
                            image_path=image_key,
                            backend="qwen_vlm",
                            detections=support_detections,
                        ),
                    )

                    if save_results:
                        # Save support panel
                        support_panel_path = (
                            viz_dir
                            / f"q{query_index:04d}_c{category_name}_{strategy_name}_{shot}shot_support_panel.png"
                        )
                        support_panel.save(support_panel_path)

                        # Save query image
                        query_image_path = (
                            viz_dir / f"q{query_index:04d}_c{category_name}_{strategy_name}_{shot}shot_query.png"
                        )
                        query_image_pil = Image.fromarray(
                            (query_image.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
                        )
                        query_image_pil.save(query_image_path)

                        query_detections_viz_path = (
                            viz_dir / f"q{query_index:04d}_c{category_name}_{strategy_name}_{shot}shot_detections.png"
                        )
                        _save_visualization(
                            image_tensor=query_image,
                            detections=support_detections,
                            backend="qwen_vlm",
                            output_path=query_detections_viz_path,
                        )

    metrics_by_method: dict[str, Any] = {}
    method_rows: list[dict[str, Any]] = []

    for method_name in sorted(method_predictions):
        predictions = method_predictions.get(method_name, [])
        ground_truth = method_ground_truth.get(method_name, [])
        if not predictions or not ground_truth:
            continue

        metrics = evaluate_dataset(predictions=predictions, ground_truth=ground_truth)
        metrics_by_method[method_name] = metrics
        method_rows.append(
            {
                "method": method_name,
                "num_eval_pairs": len(predictions),
                "ap50": float(metrics["AP50"]["ap"]),
                "ap75": float(metrics["AP75"]["ap"]),
                "mean_ap_50_75": float(metrics["mean_AP_50_75"]),
            }
        )

    run_summary = {
        "query_split": normalized_query_split,
        "support_split": normalized_support_split,
        "limit": limit,
        "requested_categories": requested_categories,
        "evaluated_query_images": total_query_images,
        "evaluated_category_pairs": total_category_pairs,
        "methods": metrics_by_method,
    }

    if save_results:
        _save_task2_results(
            run_dir=run_dir,
            metrics_by_method=run_summary,
            method_rows=method_rows,
        )

    logger.info(
        "Task 2 baseline finished on {} query image(s), {} query-category pair(s), {} method(s).",
        total_query_images,
        total_category_pairs,
        len(metrics_by_method),
    )
    for row in method_rows:
        logger.info(
            "{} -> AP50={:.4f}, AP75={:.4f}, mean_AP_50_75={:.4f}",
            row["method"],
            float(row["ap50"]),
            float(row["ap75"]),
            float(row["mean_ap_50_75"]),
        )

    return run_summary


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
    """Evaluate the selected detector backend on Task 1."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / "task1_results" / f"run_{timestamp}"

    should_create_run_dir = save_results or save_viz or qwen_debug_dump
    if should_create_run_dir:
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
    qwen_debug_entries: list[dict[str, Any]] = []

    viz_dir = run_dir / "visualizations" if save_viz else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    for index, (images, targets) in enumerate(data_loader):
        if index >= limit:
            break

        _process_single_sample(
            index=index,
            images=images,
            targets=targets,
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
    # typer.run(run_task1_baseline)
    typer.run(run_task2_support_strategy_baseline)
