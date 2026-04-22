from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import typer
from loguru import logger

from detgpt import OUTPUTS_DIR
from detgpt.box_utils import cxcywh_to_xyxy
from detgpt.data import Task1DetectionDataset
from detgpt.metrics import evaluate_dataset
from detgpt.model import QwenVLMHandler
from detgpt.support_samples import find_support_indices, side_by_side

QWEN_DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_CATEGORY_NAME = "knocker_(on_a_door)"


def _find_query_index(dataset: Task1DetectionDataset, category_name: str, query_index: int | None) -> int:
    """Resolve a query index for the requested category."""
    if query_index is not None:
        sample = dataset.samples[query_index]
        annotations = sample.get("annotations", [])
        category_names = [str(annotation.get("category_name", "")) for annotation in annotations]
        if any(name.casefold() == category_name.casefold() for name in category_names):
            return query_index
        raise ValueError(f"Query index {query_index} does not contain category '{category_name}'.")

    for sample_index, sample in enumerate(dataset.samples):
        annotations = sample.get("annotations", [])
        category_names = [str(annotation.get("category_name", "")) for annotation in annotations]
        if any(name.casefold() == category_name.casefold() for name in category_names):
            return sample_index

    raise ValueError(f"No query sample found for category '{category_name}'.")


def _save_json(data: dict[str, Any], output_path: Path) -> None:
    """Save a dictionary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(data, file_handle, indent=2)


def _prediction_record(image_path: str, detections: dict[str, Any]) -> dict[str, Any]:
    """Build one prediction record for the metrics helper."""
    return {
        "image_path": image_path,
        "boxes": detections["boxes"].detach().cpu().tolist(),
        "labels": [str(label_name) for label_name in detections["labels"]],
        "scores": detections["scores"].detach().cpu().tolist(),
    }


def _ground_truth_record(image_path: str, target: dict[str, Any], category_name: str) -> dict[str, Any]:
    """Build a filtered ground-truth record for one category."""
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


def _save_visualization(
    image_tensor: torch.Tensor,
    detections: dict[str, Any],
    output_path: Path,
) -> None:
    """Save one visualization image with predicted boxes."""
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    boxes = detections["boxes"].detach().cpu().to(torch.float32)
    scores = detections["scores"].detach().cpu().to(torch.float32)
    labels = [str(label) for label in detections["labels"]]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image_np)

    for box, score, label in zip(boxes.tolist(), scores.tolist(), labels, strict=True):
        x1, y1, x2, y2 = cxcywh_to_xyxy(box)
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


def run_text_from_vision_poc(
    category_name: str = typer.Option(
        DEFAULT_CATEGORY_NAME,
        help="LVIS category used for the support/query proof of concept.",
    ),
    split: str = typer.Option("train", help="Dataset split to use: train or val."),
    query_index: int | None = typer.Option(
        None,
        help="Optional dataset index for the query image. If omitted, the first matching sample is used.",
    ),
    n_support: int = typer.Option(1, help="Number of support examples to compose."),
    model_id: str = typer.Option(QWEN_DEFAULT_MODEL_ID, help="Qwen model id for generation and localization."),
    max_detections: int = typer.Option(1, help="Maximum detections returned by the VLM."),
    description_max_new_tokens: int = typer.Option(128, help="Maximum tokens for support description generation."),
    localization_max_new_tokens: int = typer.Option(256, help="Maximum tokens for localization generation."),
    temperature: float = typer.Option(0.0, help="Generation temperature."),
    save_viz: bool = typer.Option(
        True,
        "--save-viz/--no-save-viz",
        help="Save a query visualization with predicted boxes.",
    ),
) -> None:
    """Run a one-class VLM-only text-from-vision proof of concept."""
    normalized_split = split.strip().lower()
    if normalized_split not in {"train", "val"}:
        raise typer.BadParameter("Unsupported split. Use 'train' or 'val'.")
    if n_support < 1:
        raise typer.BadParameter("--n-support must be at least 1.")

    dataset = Task1DetectionDataset(split=normalized_split, to_float=True)
    resolved_query_index = _find_query_index(
        dataset=dataset,
        category_name=category_name,
        query_index=query_index,
    )
    support_indices = find_support_indices(
        dataset=dataset,
        category_name=category_name,
        query_index=resolved_query_index,
        n_support=n_support,
    )
    if len(support_indices) < n_support:
        raise ValueError(
            f"Requested {n_support} support sample(s) for '{category_name}', but only found {len(support_indices)}."
        )

    query_image, query_target = dataset[resolved_query_index]
    support_samples = [dataset[support_index] for support_index in support_indices]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / "text_from_vision" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    support_image = side_by_side(
        target_img=None,
        n_support_img=support_samples,
        support_category_name=category_name,
        output_path=run_dir / "support_examples.png",
    )

    vlm_handler = QwenVLMHandler(model_id=model_id)
    generated_description = vlm_handler.generate_support_description(
        support_image=support_image,
        category_name=category_name,
        max_new_tokens=description_max_new_tokens,
        temperature=temperature,
    )
    detections = vlm_handler.predict_from_description(
        image_tensor=query_image,
        description=generated_description,
        output_label=category_name,
        max_detections=max_detections,
        max_new_tokens=localization_max_new_tokens,
        temperature=temperature,
        return_debug_outputs=True,
    )

    image_path = str(dataset.samples[resolved_query_index].get("local_path", ""))
    prediction_record = _prediction_record(image_path=image_path, detections=detections)
    ground_truth_record = _ground_truth_record(
        image_path=image_path,
        target=query_target,
        category_name=category_name,
    )
    metrics = evaluate_dataset(predictions=[prediction_record], ground_truth=[ground_truth_record])

    description_path = run_dir / "generated_description.txt"
    description_path.write_text(generated_description + "\n", encoding="utf-8")
    _save_json({"predictions": [prediction_record]}, run_dir / "predictions.json")
    _save_json({"ground_truth": [ground_truth_record]}, run_dir / "ground_truth.json")
    _save_json(metrics, run_dir / "metrics.json")
    _save_json(
        {
            "category_name": category_name,
            "split": normalized_split,
            "query_index": resolved_query_index,
            "support_indices": support_indices,
            "model_id": model_id,
            "generated_description": generated_description,
            "debug_entries": detections.get("debug_entries", []),
        },
        run_dir / "run_summary.json",
    )

    if save_viz:
        _save_visualization(
            image_tensor=query_image,
            detections=detections,
            output_path=run_dir / "query_predictions.png",
        )

    logger.info("Saved text-from-vision proof of concept outputs to {}", run_dir)
    logger.info("Support description: {}", generated_description)
    logger.info("AP50={:.4f}", float(metrics["AP50"]["ap"]))


if __name__ == "__main__":
    typer.run(run_text_from_vision_poc)
