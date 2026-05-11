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
from detgpt.support_samples import (
    contextual_cropped_side_by_side,
    count_support_instances,
    cropped_side_by_side,
    find_support_indices,
    side_by_side,
)

QWEN_DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_CATEGORY_NAME = "knocker_(on_a_door)"
SUPPORT_STRATEGY_BUILDERS = {
    "side_by_side": side_by_side,
    "contextual_cropped": contextual_cropped_side_by_side,
    "cropped": cropped_side_by_side,
}


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
    support_strategy: str = typer.Option(
        "side_by_side",
        help="Support presentation strategy for description generation: side_by_side, contextual_cropped, or cropped.",
    ),
    model_id: str = typer.Option(QWEN_DEFAULT_MODEL_ID, help="Qwen model id for generation and localization."),
    max_detections: int = typer.Option(1, help="Maximum detections returned by the VLM."),
    description_max_new_tokens: int = typer.Option(128, help="Maximum tokens for support description generation."),
    localization_max_new_tokens: int = typer.Option(256, help="Maximum tokens for localization generation."),
    temperature: float = typer.Option(0.0, help="Generation temperature."),
    thinking_mode: bool = typer.Option(
        False,
        "--thinking-mode/--no-thinking-mode",
        help="Enable Qwen thinking mode while keeping parser inputs free of thinking traces.",
    ),
    thinking_max_new_tokens: int | None = typer.Option(
        512,
        help="Optional token budget for Qwen thinking before final-answer generation.",
    ),
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
    normalized_support_strategy = support_strategy.strip().lower()
    if normalized_support_strategy not in SUPPORT_STRATEGY_BUILDERS:
        supported_strategies = ", ".join(sorted(SUPPORT_STRATEGY_BUILDERS))
        raise typer.BadParameter(f"Unsupported support strategy. Use one of: {supported_strategies}.")
    if thinking_max_new_tokens is not None and thinking_max_new_tokens < 1:
        raise typer.BadParameter("--thinking-max-new-tokens must be at least 1 when provided.")

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
    support_image_count = len(support_samples)
    support_instance_count = count_support_instances(
        n_support_img=support_samples,
        support_category_name=category_name,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / "text_from_vision" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    support_builder = SUPPORT_STRATEGY_BUILDERS[normalized_support_strategy]
    support_image = support_builder(
        target_img=None,
        n_support_img=support_samples,
        support_category_name=category_name,
        output_path=run_dir / "support_examples.png",
    )

    vlm_handler = QwenVLMHandler(model_id=model_id)
    use_cropped_description_prompt = normalized_support_strategy == "cropped"
    use_contextual_cropped_description_prompt = normalized_support_strategy == "contextual_cropped"
    description_prompt_strategy = (
        "contextual_cropped"
        if use_contextual_cropped_description_prompt
        else "cropped"
        if use_cropped_description_prompt
        else "boxed"
    )
    generated_description, description_debug = vlm_handler.generate_support_description_debug(
        support_image=support_image,
        category_name=category_name,
        max_new_tokens=description_max_new_tokens,
        temperature=temperature,
        thinking_mode=thinking_mode,
        thinking_max_new_tokens=thinking_max_new_tokens,
        cropped_support=use_cropped_description_prompt,
        contextual_cropped_support=use_contextual_cropped_description_prompt,
        support_image_count=support_image_count,
        support_instance_count=support_instance_count,
    )
    detections = vlm_handler.predict_from_description(
        image_tensor=query_image,
        description=generated_description,
        output_label=category_name,
        max_detections=max_detections,
        max_new_tokens=localization_max_new_tokens,
        temperature=temperature,
        return_debug_outputs=True,
        thinking_mode=thinking_mode,
        thinking_max_new_tokens=thinking_max_new_tokens,
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
            "support_image_count": support_image_count,
            "support_instance_count": support_instance_count,
            "support_strategy": normalized_support_strategy,
            "description_prompt_strategy": description_prompt_strategy,
            "model_id": model_id,
            "thinking_mode": thinking_mode,
            "thinking_max_new_tokens": thinking_max_new_tokens,
            "generated_description": generated_description,
            "support_description_debug": description_debug,
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
