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
from detgpt.data import Task1DetectionDataset, task1_collate_fn
from detgpt.model import GroundingDINOHandler
from detgpt.visualize import _save_or_show_figure


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


def run_task1_baseline(
    save_results: bool = typer.Option(True, help="Whether to save the CSV summary."),
    save_viz: bool = typer.Option(False, help="Whether to save detection-image overlays."),
    limit: int = typer.Option(20, help="Number of samples to evaluate for testing."),
    model_id: str = typer.Option("IDEA-Research/grounding-dino-tiny", help="HF Model ID."),
) -> None:
    """
    Evaluate Grounding DINO on Task 1.
    Results are saved in a timestamped folder to prevent overwriting.
    """
    # 1. Setup timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / "task1_results" / f"run_{timestamp}"

    if save_results or save_viz:
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: {run_dir}")

    # 2. Initialize Data and Model
    dataset = Task1DetectionDataset(split="val", to_float=True)
    data_loader = DataLoader(dataset, batch_size=1, collate_fn=task1_collate_fn)
    detector = GroundingDINOHandler(model_id=model_id)

    summary_data = []

    # 3. Inference Loop
    for i, (images, targets) in enumerate(data_loader):
        if i >= limit:
            break

        image, target = images[0], targets[0]
        img_id = target["image_id"].item()
        query_categories = []
        seen_categories = set()
        for category_name in target["category_names"]:
            normalized_category_name = category_name.strip()
            if not normalized_category_name or normalized_category_name in seen_categories:
                continue
            seen_categories.add(normalized_category_name)
            query_categories.append(normalized_category_name)

        if not query_categories:
            continue

        detections = detector.predict(image, query_categories)

        # 4. Optional Visualization
        if save_viz:
            viz_dir = run_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            pred_phrases = detections.get("labels", ["object"] * len(detections["boxes"]))

            save_prediction_results(
                image=image,
                boxes=detections["boxes"],
                labels=pred_phrases,
                scores=detections["scores"],
                output_path=viz_dir / f"pred_{img_id}.png",
                title=f"DINO Zero-Shot: Image {img_id}",
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


def cxcywh_to_xyxy(box):
    cx, cy, w, h = box
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    box1 = cxcywh_to_xyxy(box1)
    box2 = cxcywh_to_xyxy(box2)

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter_area

    if union == 0:
        return 0.0

    return inter_area / union


def evaluate_image(pred, gt, iou_threshold=0.5):
    matched = set()
    tp = 0
    fp = 0

    for p_box, p_label in zip(pred["boxes"], pred["labels"], strict=False):
        found_match = False

        for i, (g_box, g_label) in enumerate(zip(gt["boxes"], gt["labels"], strict=False)):
            if i in matched:
                continue

            if p_label != g_label:
                continue

            iou = compute_iou(p_box, g_box)

            if iou >= iou_threshold:
                tp += 1
                matched.add(i)
                found_match = True
                break

        if not found_match:
            fp += 1

    fn = len(gt["boxes"]) - len(matched)

    return {"tp": tp, "fp": fp, "fn": fn}


def compute_precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall


def evaluate_dataset(predictions, ground_truth):
    total_tp = total_fp = total_fn = 0

    for pred, gt in zip(predictions, ground_truth, strict=False):
        res = evaluate_image(pred, gt)
        total_tp += res["tp"]
        total_fp += res["fp"]
        total_fn += res["fn"]

    precision, recall = compute_precision_recall(total_tp, total_fp, total_fn)

    return {
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


if __name__ == "__main__":
    predictions = [
        {
            "boxes": [[142, 92, 192, 95]],
            "labels": ["car"],
        }
    ]

    ground_truth = [
        {
            "boxes": [[140, 90, 190, 100]],
            "labels": ["car"],
        }
    ]

    results = evaluate_dataset(predictions, ground_truth)
    print(results)
