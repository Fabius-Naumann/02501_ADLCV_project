from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLOWorld

from detgpt import OUTPUTS_DIR
from detgpt.box_utils import xyxy_to_cxcywh


def predict_image(model: YOLOWorld, image_path: Path, class_names: list[str]) -> dict:
    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.05,
        verbose=False,
    )

    result = results[0]
    boxes_xyxy = result.boxes.xyxy.cpu().tolist()
    scores = result.boxes.conf.cpu().tolist()
    label_ids = [int(x) for x in result.boxes.cls.cpu().tolist()]

    boxes_cxcywh = [xyxy_to_cxcywh(box) for box in boxes_xyxy]
    labels = [class_names[i] for i in label_ids]

    return {
        "image_path": str(image_path),
        "boxes": boxes_cxcywh,
        "scores": scores,
        "labels": labels,
    }


def run_yolo_world_on_folder(image_dir: str | Path, class_names: list[str]) -> list[dict]:
    image_dir = Path(image_dir)
    image_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))

    model = YOLOWorld("yolov8s-world.pt")
    model.set_classes(class_names)

    output_dir = OUTPUTS_DIR
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    all_predictions = []

    for image_path in image_paths:
        prediction = predict_image(model, image_path, class_names)
        all_predictions.append(prediction)

    with (predictions_dir / "yolo_world_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2)

    print(f"Saved predictions for {len(all_predictions)} images.")
    return all_predictions


if __name__ == "__main__":
    run_yolo_world_on_folder(
        image_dir="data",
        class_names=["person", "car", "bicycle", "bus", "truck"],
    )
