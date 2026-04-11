import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import matplotlib.pyplot as plt
import torch
import typer
from loguru import logger
from matplotlib.patches import Rectangle
from torch import Tensor
from torch.utils.data import DataLoader

from detgpt import OUTPUTS_DIR
from detgpt.box_utils import cxcywh_tensor_to_xyxy
from detgpt.data import Task1DetectionDataset, task1_collate_fn
from detgpt.model import GroundingDINOHandler, InternVLHandler, QwenVLMHandler
from detgpt.visualize import _save_or_show_figure

DINO_DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
QWEN_DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
INTERNVL_DEFAULT_MODEL_ID = "OpenGVLab/InternVL2_5-8B"


def save_prediction_results(
    image: Tensor, boxes: Tensor, labels: list[str], scores: Tensor, output_path: Path, title: str = "Model Predictions"
) -> None:
    """
    Renders and saves an image with predicted bounding boxes (xyxy format).

    Args:
        image: C x H x W tensor.
        boxes: N x 4 tensor of absolute [xmin, ymin, xmax, ymax].
        labels: List of N strings (category names).
        scores: N tensor of confidence scores.
        output_path: Path to save the resulting .png.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    # Convert tensor to [H, W, C] for matplotlib
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())

    for box, label, score in zip(boxes, labels, scores, strict=False):
        xmin, ymin, xmax, ymax = box.tolist()
        width, height = xmax - xmin, ymax - ymin
        score_value = score.item()

        rect = Rectangle((xmin, ymin), width, height, edgecolor="lime", facecolor="none", linewidth=2)
        ax.add_patch(rect)

        # Add label with confidence score
        ax.text(
            xmin,
            ymin - 5,
            f"{label}: {score_value:.2f}",
            color="white",
            fontsize=10,
            bbox={"facecolor": "lime", "alpha": 0.5},
        )

    ax.set_title(title)
    ax.axis("off")
    _save_or_show_figure(fig, output_path)


def _resolve_detector(
    detector_backend: str,
    model_id: str | None,
) -> tuple[str, str, GroundingDINOHandler | QwenVLMHandler | InternVLHandler]:
    """Resolve backend name, model id, and detector instance."""
    normalized_backend = detector_backend.strip().lower()

    if normalized_backend == "grounding_dino":
        resolved_model_id = model_id or DINO_DEFAULT_MODEL_ID
        return normalized_backend, resolved_model_id, GroundingDINOHandler(model_id=resolved_model_id)

    if normalized_backend == "qwen_vlm":
        resolved_model_id = model_id or QWEN_DEFAULT_MODEL_ID
        return normalized_backend, resolved_model_id, QwenVLMHandler(model_id=resolved_model_id)

    if normalized_backend == "intern_vl":
        resolved_model_id = model_id or INTERNVL_DEFAULT_MODEL_ID
        return normalized_backend, resolved_model_id, InternVLHandler(model_id=resolved_model_id)

    raise typer.BadParameter(
        f"Unsupported detector backend '{detector_backend}'. Use 'grounding_dino', 'qwen_vlm', or 'intern_vl'."
    )


def _extract_query_categories(category_names: list[str]) -> list[str]:
    """Build unique non-empty category list preserving order."""
    query_categories: list[str] = []
    seen_categories: set[str] = set()
    for category_name in category_names:
        normalized_category_name = category_name.strip()
        if not normalized_category_name or normalized_category_name in seen_categories:
            continue
        seen_categories.add(normalized_category_name)
        query_categories.append(normalized_category_name)
    return query_categories


def _predict_with_backend(
    detector: GroundingDINOHandler | QwenVLMHandler | InternVLHandler,
    normalized_backend: str,
    image: Tensor,
    query_categories: list[str],
    qwen_max_detections_per_category: int,
    qwen_temperature: float,
    qwen_return_debug_outputs: bool,
) -> dict[str, Tensor | list[str] | list[dict[str, Any]]]:
    """Run backend-specific prediction logic."""
    if normalized_backend == "qwen_vlm":
        return detector.predict(
            image,
            query_categories,
            max_detections_per_category=qwen_max_detections_per_category,
            temperature=qwen_temperature,
            return_debug_outputs=qwen_return_debug_outputs,
        )

    if normalized_backend == "intern_vl":
        return detector.predict(
            image,
            query_categories,
            max_detections_per_category=qwen_max_detections_per_category,
            return_debug_outputs=qwen_return_debug_outputs,
        )

    return detector.predict(image, query_categories)


def _boxes_for_visualization(normalized_backend: str, boxes: Tensor) -> Tensor:
    """Return boxes in xyxy format for rendering."""
    if normalized_backend not in {"qwen_vlm", "intern_vl"}:
        return boxes

    return cxcywh_tensor_to_xyxy(boxes)


def _open_qwen_debug_trace_file(
    run_dir: Path,
    normalized_backend: str,
    qwen_debug_dump: bool,
) -> tuple[Path | None, TextIO | None]:
    """Open qwen debug trace file when enabled."""
    if normalized_backend != "qwen_vlm" or not qwen_debug_dump:
        return None, None

    run_dir.mkdir(parents=True, exist_ok=True)
    debug_trace_path = run_dir / "qwen_debug_trace.jsonl"
    debug_file_handle = debug_trace_path.open("w", encoding="utf-8")
    logger.info(f"Qwen debug trace enabled: {debug_trace_path}")
    return debug_trace_path, debug_file_handle


def _safe_tensor_to_list(value: Tensor | list[str] | list[dict[str, Any]] | None) -> Any:
    """Convert tensors to lists while keeping other values unchanged."""
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    return value


def _build_qwen_debug_record(
    img_id: int,
    image_file: str | None,
    query_categories: list[str],
    detections: dict[str, Tensor | list[str] | list[dict[str, Any]]],
) -> dict[str, Any]:
    """Create structured record with input/raw/parsed/final qwen outputs."""
    debug_entries_any = detections.get("debug_entries", [])
    debug_entries = debug_entries_any if isinstance(debug_entries_any, list) else []

    input_prompts: list[dict[str, str]] = []
    raw_outputs: list[dict[str, str]] = []
    parsed_outputs: list[dict[str, Any]] = []
    final_outputs_per_category: list[dict[str, Any]] = []

    for entry in debug_entries:
        if not isinstance(entry, dict):
            continue

        category_name = str(entry.get("category_name", ""))
        input_prompts.append(
            {
                "category_name": category_name,
                "prompt": str(entry.get("input_prompt", "")),
            }
        )
        raw_outputs.append(
            {
                "category_name": category_name,
                "text": str(entry.get("raw_output_text", "")),
            }
        )
        parsed_outputs.append(
            {
                "category_name": category_name,
                "parsed": entry.get("parsed_output", {}),
            }
        )
        final_outputs_per_category.append(
            {
                "category_name": category_name,
                "final": entry.get("final_output", {}),
            }
        )

    return {
        "image_id": img_id,
        "image_file": image_file,
        "query_categories": query_categories,
        "input": {
            "prompts": input_prompts,
        },
        "raw_output": raw_outputs,
        "parsed_output": parsed_outputs,
        "final_output": {
            "combined": {
                "boxes_cxcywh": _safe_tensor_to_list(detections.get("boxes")),
                "scores": _safe_tensor_to_list(detections.get("scores")),
                "labels": _safe_tensor_to_list(detections.get("labels")),
            },
            "per_category": final_outputs_per_category,
        },
    }


def _try_get_image_file(dataset: Task1DetectionDataset, index: int) -> str | None:
    """Resolve image local path for a sample index."""
    if index < 0 or index >= len(dataset.samples):
        return None

    sample = dataset.samples[index]
    local_path = sample.get("local_path")
    if isinstance(local_path, str) and local_path:
        return local_path
    return None


def _write_qwen_debug_record(
    debug_file_handle: TextIO | None,
    img_id: int,
    image_file: str | None,
    query_categories: list[str],
    detections: dict[str, Tensor | list[str] | list[dict[str, Any]]],
) -> None:
    """Write one qwen debug record to JSONL file."""
    if debug_file_handle is None:
        return

    record = _build_qwen_debug_record(
        img_id=img_id,
        image_file=image_file,
        query_categories=query_categories,
        detections=detections,
    )
    debug_file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _run_inference_loop(
    data_loader: DataLoader,
    dataset: Task1DetectionDataset,
    detector: GroundingDINOHandler | QwenVLMHandler | InternVLHandler,
    normalized_backend: str,
    qwen_max_detections_per_category: int,
    qwen_temperature: float,
    qwen_debug_dump: bool,
    debug_file_handle: TextIO | None,
    save_viz: bool,
    save_results: bool,
    run_dir: Path,
    limit: int,
) -> list[dict[str, Any]]:
    """Run inference loop and return summary rows."""
    summary_data: list[dict[str, Any]] = []

    for i, (images, targets) in enumerate(data_loader):
        if i >= limit:
            break

        image, target = images[0], targets[0]
        img_id = target["image_id"].item()
        query_categories = _extract_query_categories(target["category_names"])

        if not query_categories:
            continue

        detections = _predict_with_backend(
            detector=detector,
            normalized_backend=normalized_backend,
            image=image,
            query_categories=query_categories,
            qwen_max_detections_per_category=qwen_max_detections_per_category,
            qwen_temperature=qwen_temperature,
            qwen_return_debug_outputs=qwen_debug_dump,
        )

        if normalized_backend == "qwen_vlm" and qwen_debug_dump:
            _write_qwen_debug_record(
                debug_file_handle=debug_file_handle,
                img_id=img_id,
                image_file=_try_get_image_file(dataset=dataset, index=i),
                query_categories=query_categories,
                detections=detections,
            )

        if save_viz:
            viz_dir = run_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            pred_phrases = [str(label) for label in detections.get("labels", ["object"] * len(detections["boxes"]))]
            viz_boxes = _boxes_for_visualization(normalized_backend, detections["boxes"])

            save_prediction_results(
                image=image,
                boxes=viz_boxes,
                labels=pred_phrases,
                scores=detections["scores"],
                output_path=viz_dir / f"pred_{img_id}.png",
                title=f"{normalized_backend} Zero-Shot: Image {img_id}",
            )

        if save_results:
            summary_data.append(
                {
                    "image_id": img_id,
                    "num_gt": len(target["boxes"]),
                    "num_pred": len(detections["boxes"]),
                    "categories": "|".join(query_categories),
                    "avg_score": torch.mean(detections["scores"]).item() if len(detections["scores"]) > 0 else 0,
                }
            )

    return summary_data


def run_task1_baseline(
    save_results: bool = typer.Option(True, help="Whether to save the CSV summary."),
    save_viz: bool = typer.Option(False, help="Whether to save detection-image overlays."),
    split: str = typer.Option("val", help="Dataset split to evaluate: val or train."),
    limit: int = typer.Option(20, help="Number of samples to evaluate for testing."),
    detector_backend: str = typer.Option(
        "grounding_dino",
        help="Detector backend: grounding_dino, qwen_vlm, or intern_vl.",
    ),
    model_id: str | None = typer.Option(
        None,
        help="HF Model ID. If omitted, a backend-specific default is used.",
    ),
    qwen_max_detections_per_category: int = typer.Option(
        1,
        help="Maximum detections returned per queried class for qwen_vlm.",
    ),
    qwen_temperature: float = typer.Option(
        0.0,
        help="Decoding temperature for qwen_vlm. Use 0.0 for deterministic output.",
    ),
    qwen_debug_dump: bool = typer.Option(
        False,
        help="Write qwen debug trace with input/raw/parsed/final outputs to JSONL file.",
    ),
) -> None:
    """
    Evaluate selected detector backend on Task 1.
    Results are saved in a timestamped folder to prevent overwriting.
    """
    # 1. Setup timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / "task1_results" / f"run_{timestamp}"

    if save_results or save_viz or qwen_debug_dump:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {run_dir}")

    # 2. Initialize Data and Model
    normalized_split = split.strip().lower()
    if normalized_split not in {"val", "train"}:
        raise typer.BadParameter("Unsupported split. Use 'val' or 'train'.")

    dataset = Task1DetectionDataset(split=normalized_split, to_float=True)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=task1_collate_fn)
    normalized_backend, resolved_model_id, detector = _resolve_detector(
        detector_backend=detector_backend,
        model_id=model_id,
    )

    debug_trace_path, debug_file_handle = _open_qwen_debug_trace_file(
        run_dir=run_dir,
        normalized_backend=normalized_backend,
        qwen_debug_dump=qwen_debug_dump,
    )

    logger.info(f"Using split={normalized_split}, backend={normalized_backend}, model_id={resolved_model_id}")

    try:
        summary_data = _run_inference_loop(
            data_loader=data_loader,
            dataset=dataset,
            detector=detector,
            normalized_backend=normalized_backend,
            qwen_max_detections_per_category=qwen_max_detections_per_category,
            qwen_temperature=qwen_temperature,
            qwen_debug_dump=qwen_debug_dump,
            debug_file_handle=debug_file_handle,
            save_viz=save_viz,
            save_results=save_results,
            run_dir=run_dir,
            limit=limit,
        )
    finally:
        if debug_file_handle is not None:
            debug_file_handle.close()
            if debug_trace_path is not None:
                logger.info(f"Qwen debug trace saved to {debug_trace_path}")

    # 6. Finalize Results
    if save_results and summary_data:
        csv_path = run_dir / "detections_summary.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "num_gt", "num_pred", "categories", "avg_score"])
            writer.writeheader()
            writer.writerows(summary_data)
        logger.info(f"Summary saved to {csv_path}")


if __name__ == "__main__":
    typer.run(run_task1_baseline)

    # predictions = [
    #     {
    #         "boxes": [[142, 92, 192, 95]],
    #         "labels": ["car"],
    #     }
    # ]

    # ground_truth = [
    #     {
    #         "boxes": [[140, 90, 190, 100]],
    #         "labels": ["car"],
    #     }
    # ]

    # results = evaluate_dataset(predictions, ground_truth)
    # print(results)
