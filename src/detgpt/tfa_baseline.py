from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import typer
from loguru import logger
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from detgpt import OUTPUTS_DIR
from detgpt.metrics import evaluate_dataset


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Manifest at {manifest_path} must contain a list of samples.")

    return data


def _extract_class_names_from_support(
    support_manifest: list[dict[str, Any]],
) -> list[str]:
    """
    Define the FSOD label space strictly from the support set.

    This avoids accidentally expanding the benchmark with unrelated annotations
    that happen to exist in query images.
    """
    seen: set[str] = set()
    class_names: list[str] = []

    for sample in support_manifest:
        for annotation in sample.get("annotations", []):
            class_name = str(annotation.get("category_name", "")).strip()
            if not class_name or class_name in seen:
                continue
            seen.add(class_name)
            class_names.append(class_name)

    return class_names


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = box
    return [x, y, x + w, y + h]


def _xyxy_to_cxcywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return [center_x, center_y, width, height]


class ManifestDetectionDataset(Dataset):
    """
    Dataset for support/query manifests produced by fsod_data.py.

    The dataset filters annotations to the class names provided through
    class_to_id, so query images only contribute target objects from the
    intended FSOD label space.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        class_to_id: dict[str, int],
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.samples = _load_manifest(self.manifest_path)
        self.class_to_id = class_to_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        sample = self.samples[index]

        image_path = str(sample.get("local_path", "")).strip()
        if not image_path:
            raise ValueError(f"Sample at index {index} is missing local_path.")

        image = read_image(image_path, mode=ImageReadMode.RGB).to(torch.float32) / 255.0

        boxes_xyxy: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        category_names: list[str] = []

        for annotation in sample.get("annotations", []):
            class_name = str(annotation.get("category_name", "")).strip()
            if not class_name:
                continue
            if class_name not in self.class_to_id:
                continue

            bbox_xywh = annotation.get("bbox_xywh")
            if not isinstance(bbox_xywh, list) or len(bbox_xywh) != 4:
                continue

            xyxy_box = _xywh_to_xyxy([float(v) for v in bbox_xywh])
            boxes_xyxy.append(xyxy_box)
            labels.append(int(self.class_to_id[class_name]))
            areas.append(float(annotation.get("area", 0.0) or 0.0))
            category_names.append(class_name)

        if boxes_xyxy:
            boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            areas_tensor = torch.tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.zeros((len(boxes_xyxy),), dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
            "image_id": torch.tensor([int(sample["image_id"])], dtype=torch.int64),
            "category_names": category_names,
            "image_path": image_path,
        }
        return image, target


def detection_collate_fn(
    batch: list[tuple[torch.Tensor, dict[str, Any]]],
) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Build Faster R-CNN and replace the final predictor.
    num_classes includes background.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def freeze_for_tfa_head_only(model: nn.Module) -> None:
    """
    Simplified TFA-style setup:
    freeze everything except the final RoI box predictor.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.roi_heads.box_predictor.parameters():
        parameter.requires_grad = True


def train_tfa_head(
    model: nn.Module,
    support_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> None:
    model.to(device)
    model.train()

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Did you freeze everything accidentally?")

    optimizer = SGD(
        trainable_params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for images, targets in support_loader:
            images = [image.to(device) for image in images]

            moved_targets: list[dict[str, Any]] = []
            for target in targets:
                moved_targets.append(
                    {
                        "boxes": target["boxes"].to(device),
                        "labels": target["labels"].to(device),
                        "area": target["area"].to(device),
                        "iscrowd": target["iscrowd"].to(device),
                        "image_id": target["image_id"].to(device),
                    }
                )

            loss_dict = model(images, moved_targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            num_batches += 1

        mean_loss = epoch_loss / max(num_batches, 1)
        logger.info("Epoch {}/{} | support loss = {:.6f}", epoch + 1, epochs, mean_loss)


@torch.no_grad()
def predict_query_set(
    model: nn.Module,
    query_loader: DataLoader,
    device: torch.device,
    id_to_class: dict[int, str],
    score_threshold: float,
    debug_predictions: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.to(device)
    model.eval()

    predictions: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []

    num_images_with_raw_outputs = 0
    total_raw_boxes = 0
    total_kept_boxes = 0

    for image_index, (images, targets) in enumerate(query_loader):
        images_on_device = [image.to(device) for image in images]
        outputs = model(images_on_device)

        for target, output in zip(targets, outputs, strict=True):
            image_path = str(target["image_path"])

            gt_boxes_cxcywh = [_xyxy_to_cxcywh(box) for box in target["boxes"].cpu().tolist()]
            gt_labels = [str(label) for label in target["category_names"]]

            ground_truth.append(
                {
                    "image_path": image_path,
                    "boxes": gt_boxes_cxcywh,
                    "labels": gt_labels,
                }
            )

            pred_boxes_cxcywh: list[list[float]] = []
            pred_labels: list[str] = []
            pred_scores: list[float] = []

            output_boxes = output["boxes"].detach().cpu().tolist()
            output_labels = output["labels"].detach().cpu().tolist()
            output_scores = output["scores"].detach().cpu().tolist()

            raw_count = len(output_boxes)
            total_raw_boxes += raw_count
            if raw_count > 0:
                num_images_with_raw_outputs += 1

            if debug_predictions and image_index < 5:
                logger.info(
                    "[debug] image={} raw_boxes={} top_scores={} top_labels={}",
                    image_index,
                    raw_count,
                    output_scores[:10],
                    output_labels[:10],
                )

            for box_xyxy, label_id, score in zip(output_boxes, output_labels, output_scores, strict=True):
                if float(score) < score_threshold:
                    continue

                class_name = id_to_class.get(int(label_id))
                if class_name is None:
                    continue

                pred_boxes_cxcywh.append(_xyxy_to_cxcywh([float(v) for v in box_xyxy]))
                pred_labels.append(class_name)
                pred_scores.append(float(score))

            total_kept_boxes += len(pred_boxes_cxcywh)

            predictions.append(
                {
                    "image_path": image_path,
                    "boxes": pred_boxes_cxcywh,
                    "labels": pred_labels,
                    "scores": pred_scores,
                }
            )

    logger.info(
        "Prediction summary | images_with_raw_outputs={} total_raw_boxes={} total_kept_boxes={} threshold={}",
        num_images_with_raw_outputs,
        total_raw_boxes,
        total_kept_boxes,
        score_threshold,
    )

    return predictions, ground_truth


def _save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def run_tfa_baseline(
    support_manifest_path: str = typer.Option(..., help="Path to the support manifest JSON."),
    query_manifest_path: str = typer.Option(..., help="Path to the query manifest JSON."),
    epochs: int = typer.Option(15, help="Number of support fine-tuning epochs."),
    lr: float = typer.Option(0.002, help="Learning rate for TFA head fine-tuning."),
    momentum: float = typer.Option(0.9, help="SGD momentum."),
    weight_decay: float = typer.Option(1e-4, help="Weight decay."),
    batch_size: int = typer.Option(2, help="Support-set batch size."),
    score_threshold: float = typer.Option(0.05, help="Prediction score threshold."),
    pretrained: bool = typer.Option(True, "--pretrained/--random-init", help="Use COCO-pretrained detector."),
    save_results: bool = typer.Option(True, "--save-results/--no-save-results", help="Save JSON outputs."),
    device_str: str | None = typer.Option(None, help="Force device: cuda or cpu."),
    debug_predictions: bool = typer.Option(
        False,
        "--debug-predictions/--no-debug-predictions",
        help="Log raw prediction counts/scores for the first few query images.",
    ),
) -> None:
    """
    Windows-native simplified TFA-style FSOD baseline:
    - load support/query manifests
    - define label space from support classes only
    - build pretrained Faster R-CNN
    - freeze all except box predictor
    - fine-tune on support set
    - evaluate on query set with detgpt.metrics.evaluate_dataset
    """
    support_manifest = _load_manifest(support_manifest_path)
    _ = _load_manifest(query_manifest_path)  # validate path exists

    class_names = _extract_class_names_from_support(support_manifest)
    if not class_names:
        raise ValueError("No class names found in the support manifest.")

    class_to_id = {class_name: index + 1 for index, class_name in enumerate(class_names)}
    id_to_class = {index: class_name for class_name, index in class_to_id.items()}

    support_dataset = ManifestDetectionDataset(
        manifest_path=support_manifest_path,
        class_to_id=class_to_id,
    )
    query_dataset = ManifestDetectionDataset(
        manifest_path=query_manifest_path,
        class_to_id=class_to_id,
    )

    support_loader = DataLoader(
        support_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=detection_collate_fn,
    )

    if device_str is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device_str)

    logger.info("Using device: {}", resolved_device)
    logger.info("Classes from support only: {}", class_names)
    logger.info("Support samples: {}", len(support_dataset))
    logger.info("Query samples: {}", len(query_dataset))

    model = build_model(
        num_classes=len(class_names) + 1,
        pretrained=pretrained,
    )
    freeze_for_tfa_head_only(model)

    train_tfa_head(
        model=model,
        support_loader=support_loader,
        device=resolved_device,
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    predictions, ground_truth = predict_query_set(
        model=model,
        query_loader=query_loader,
        device=resolved_device,
        id_to_class=id_to_class,
        score_threshold=score_threshold,
        debug_predictions=debug_predictions,
    )

    metrics = evaluate_dataset(predictions=predictions, ground_truth=ground_truth)

    logger.info(
        "AP50={:.4f}, AP75={:.4f}, mean_AP_50_75={:.4f}",
        float(metrics["AP50"]["ap"]),
        float(metrics["AP75"]["ap"]),
        float(metrics["mean_AP_50_75"]),
    )

    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUTS_DIR / "tfa_baseline" / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        _save_json(predictions, run_dir / "predictions.json")
        _save_json(ground_truth, run_dir / "ground_truth.json")
        _save_json(metrics, run_dir / "metrics.json")

        config_payload = {
            "support_manifest_path": str(Path(support_manifest_path).resolve()),
            "query_manifest_path": str(Path(query_manifest_path).resolve()),
            "epochs": epochs,
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "score_threshold": score_threshold,
            "pretrained": pretrained,
            "device": str(resolved_device),
            "classes": class_names,
            "debug_predictions": debug_predictions,
        }
        _save_json(config_payload, run_dir / "run_config.json")

        logger.info("Saved results to {}", run_dir)


if __name__ == "__main__":
    typer.run(run_tfa_baseline)
