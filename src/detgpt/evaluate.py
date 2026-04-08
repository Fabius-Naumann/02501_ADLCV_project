import csv
from datetime import datetime
from pathlib import Path

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
from detgpt.model import GroundingDINOHandler, QwenVLMHandler
from detgpt.visualize import _save_or_show_figure

DINO_DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
QWEN_DEFAULT_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


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
) -> tuple[str, str, GroundingDINOHandler | QwenVLMHandler]:
    """Resolve backend name, model id, and detector instance."""
    normalized_backend = detector_backend.strip().lower()

    if normalized_backend == "grounding_dino":
        resolved_model_id = model_id or DINO_DEFAULT_MODEL_ID
        return normalized_backend, resolved_model_id, GroundingDINOHandler(model_id=resolved_model_id)

    if normalized_backend == "qwen_vlm":
        resolved_model_id = model_id or QWEN_DEFAULT_MODEL_ID
        return normalized_backend, resolved_model_id, QwenVLMHandler(model_id=resolved_model_id)

    raise typer.BadParameter(f"Unsupported detector backend '{detector_backend}'. Use 'grounding_dino' or 'qwen_vlm'.")


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
    detector: GroundingDINOHandler | QwenVLMHandler,
    normalized_backend: str,
    image: Tensor,
    query_categories: list[str],
    qwen_max_detections_per_category: int,
    qwen_temperature: float,
    qwen_return_raw_text_outputs: bool,
) -> dict[str, Tensor | list[str] | dict[str, str]]:
    """Run backend-specific prediction logic."""
    if normalized_backend == "qwen_vlm":
        return detector.predict(
            image,
            query_categories,
            max_detections_per_category=qwen_max_detections_per_category,
            temperature=qwen_temperature,
            return_raw_text_outputs=qwen_return_raw_text_outputs,
        )

    return detector.predict(image, query_categories)


def _boxes_for_visualization(normalized_backend: str, boxes: Tensor) -> Tensor:
    """Return boxes in xyxy format for rendering."""
    if normalized_backend != "qwen_vlm":
        return boxes

    return cxcywh_tensor_to_xyxy(boxes)


def _log_qwen_raw_outputs(
    img_id: int,
    detections: dict[str, Tensor | list[str] | dict[str, str]],
) -> None:
    """Log raw text outputs captured from qwen_vlm for debugging."""
    raw_text_outputs = detections.get("raw_text_outputs", {})
    if not isinstance(raw_text_outputs, dict) or not raw_text_outputs:
        logger.warning(f"Image {img_id}: no raw qwen text outputs were captured.")
        return

    for category_name, raw_text in raw_text_outputs.items():
        logger.info(f"Image {img_id} | Category '{category_name}' | Raw Qwen output:\n{raw_text}")


def run_task1_baseline(
    save_results: bool = typer.Option(True, help="Whether to save the CSV summary."),
    save_viz: bool = typer.Option(False, help="Whether to save detection-image overlays."),
    limit: int = typer.Option(20, help="Number of samples to evaluate for testing."),
    detector_backend: str = typer.Option(
        "grounding_dino",
        help="Detector backend: grounding_dino or qwen_vlm.",
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
    qwen_log_raw_text: bool = typer.Option(
        False,
        help="Log raw generated text per category for qwen_vlm debugging.",
    ),
) -> None:
    """
    Evaluate selected detector backend on Task 1.
    Results are saved in a timestamped folder to prevent overwriting.
    """
    # 1. Setup timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / "task1_results" / f"run_{timestamp}"

    if save_results or save_viz:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {run_dir}")

    # 2. Initialize Data and Model
    dataset = Task1DetectionDataset(split="train", to_float=True)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=task1_collate_fn)
    normalized_backend, resolved_model_id, detector = _resolve_detector(
        detector_backend=detector_backend,
        model_id=model_id,
    )

    logger.info(f"Using backend={normalized_backend}, model_id={resolved_model_id}")

    summary_data = []

    # 3. Inference Loop
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
            qwen_return_raw_text_outputs=qwen_log_raw_text,
        )

        if normalized_backend == "qwen_vlm" and qwen_log_raw_text:
            _log_qwen_raw_outputs(img_id=img_id, detections=detections)

        # 4. Optional Visualization
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

        # 5. Collect Data
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
