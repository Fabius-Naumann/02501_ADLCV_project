from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from detgpt.box_utils import xywh_to_cxcywh


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path) -> None:
    """Save JSON to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def build_image_id_to_path(coco_gt: dict[str, Any]) -> dict[int, str]:
    """Build image_id -> file_name mapping from COCO-style GT JSON."""
    images = coco_gt.get("images", [])
    return {int(image["id"]): str(image["file_name"]) for image in images}


def build_category_id_to_name(coco_gt: dict[str, Any]) -> dict[int, str]:
    """Build category_id -> class name mapping from COCO-style GT JSON."""
    categories = coco_gt.get("categories", [])
    return {int(category["id"]): str(category["name"]) for category in categories}


def convert_defrcn_predictions_to_eval_format(
    coco_predictions: list[dict[str, Any]],
    coco_gt: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert COCO-style predictions to the project's evaluation JSON format.
    """
    image_id_to_path = build_image_id_to_path(coco_gt)
    category_id_to_name = build_category_id_to_name(coco_gt)

    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "image_path": "",
            "boxes": [],
            "labels": [],
            "scores": [],
        }
    )

    for prediction in coco_predictions:
        image_id = int(prediction["image_id"])
        category_id = int(prediction["category_id"])
        bbox_xywh = [float(value) for value in prediction["bbox"]]
        score = float(prediction["score"])

        image_path = image_id_to_path[image_id]
        class_name = category_id_to_name[category_id]

        grouped_entry = grouped[image_path]
        grouped_entry["image_path"] = image_path
        grouped_entry["boxes"].append(xywh_to_cxcywh(bbox_xywh))
        grouped_entry["labels"].append(class_name)
        grouped_entry["scores"].append(score)

    return list(grouped.values())


def save_eval_predictions_from_defrcn(
    predictions_json_path: str | Path,
    coco_gt_json_path: str | Path,
    output_eval_json_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Convert DeFRCN raw COCO predictions into project eval JSON and save them.
    """
    coco_predictions = load_json(predictions_json_path)
    coco_gt = load_json(coco_gt_json_path)

    if not isinstance(coco_predictions, list):
        raise ValueError("DeFRCN predictions JSON must be a list of prediction objects.")
    if not isinstance(coco_gt, dict):
        raise ValueError("COCO ground-truth JSON must be a dictionary.")

    eval_predictions = convert_defrcn_predictions_to_eval_format(
        coco_predictions=coco_predictions,
        coco_gt=coco_gt,
    )
    save_json(eval_predictions, output_eval_json_path)
    return eval_predictions


def run_defrcn_train(
    defrcn_root: str | Path,
    config_file: str | Path,
    num_gpus: int = 1,
    extra_opts: list[str] | None = None,
) -> None:
    """
    Run DeFRCN training using the official repo entrypoint.
    """
    root = Path(defrcn_root)
    command = [
        "python",
        "main.py",
        "--num-gpus",
        str(num_gpus),
        "--config-file",
        str(config_file),
    ]

    if extra_opts:
        command.extend(extra_opts)

    subprocess.run(command, cwd=root, check=True)


def run_defrcn_eval(
    defrcn_root: str | Path,
    config_file: str | Path,
    num_gpus: int = 1,
    extra_opts: list[str] | None = None,
) -> None:
    """
    Run DeFRCN evaluation using the official repo entrypoint.
    """
    root = Path(defrcn_root)
    command = [
        "python",
        "main.py",
        "--num-gpus",
        str(num_gpus),
        "--config-file",
        str(config_file),
        "--eval-only",
    ]

    if extra_opts:
        command.extend(extra_opts)

    subprocess.run(command, cwd=root, check=True)
