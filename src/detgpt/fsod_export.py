from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from detgpt.box_utils import cxcywh_to_xyxy
from detgpt.lvis_api import default_manifest_path


def _load_manifest_from_path(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Load a manifest from an explicit path."""
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Manifest JSON must contain a list of samples.")

    return data


def _load_manifest_from_split(split: str) -> list[dict[str, Any]]:
    """Load the default manifest for a split."""
    manifest_path = default_manifest_path(split)
    return _load_manifest_from_path(manifest_path)


def _centered_dict_to_cxcywh(centered: dict[str, Any]) -> list[float]:
    """Convert bbox_xywh_centered dict to [cx, cy, w, h]."""
    return [
        float(centered.get("x_center", 0.0)),
        float(centered.get("y_center", 0.0)),
        float(centered.get("width", 0.0)),
        float(centered.get("height", 0.0)),
    ]


def _annotation_to_coco_bbox(annotation: dict[str, Any]) -> list[float]:
    """
    Export bbox as COCO xywh using the stored original bbox if available.

    Fallback: derive xywh from centered box.
    """
    bbox_xywh = annotation.get("bbox_xywh")
    if isinstance(bbox_xywh, list) and len(bbox_xywh) == 4:
        return [float(value) for value in bbox_xywh]

    centered = annotation.get("bbox_xywh_centered", {})
    if isinstance(centered, dict):
        cxcywh = _centered_dict_to_cxcywh(centered)
        x1, y1, x2, y2 = cxcywh_to_xyxy(cxcywh)
        width = x2 - x1
        height = y2 - y1
        return [x1, y1, width, height]

    return [0.0, 0.0, 0.0, 0.0]


def _export_manifest_records_to_coco(
    manifest: list[dict[str, Any]],
    output_path: str | Path,
    class_names: list[str] | None = None,
) -> None:
    """Export manifest records to COCO-style JSON."""
    allowed = set(class_names) if class_names else None

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    categories_by_name: dict[str, int] = {}
    next_category_id = 1

    for sample in manifest:
        image_id = int(sample["image_id"])
        local_path = str(sample.get("local_path", ""))

        images.append(
            {
                "id": image_id,
                "file_name": local_path,
                "width": int(sample.get("width", 0) or 0),
                "height": int(sample.get("height", 0) or 0),
            }
        )

        for ann in sample.get("annotations", []):
            class_name = str(ann.get("category_name", "")).strip()
            if allowed is not None and class_name not in allowed:
                continue
            if not class_name:
                continue

            if class_name not in categories_by_name:
                categories_by_name[class_name] = next_category_id
                next_category_id += 1

            annotations.append(
                {
                    "id": int(ann.get("annotation_id", len(annotations) + 1)),
                    "image_id": image_id,
                    "category_id": categories_by_name[class_name],
                    "bbox": _annotation_to_coco_bbox(ann),
                    "area": float(ann.get("area", 0.0) or 0.0),
                    "iscrowd": 0,
                }
            )

    categories = [{"id": category_id, "name": class_name} for class_name, category_id in categories_by_name.items()]

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def _export_manifest_records_to_eval_json(
    manifest: list[dict[str, Any]],
    output_path: str | Path,
    class_names: list[str] | None = None,
) -> None:
    """Export manifest records to the project's eval JSON format."""
    allowed = set(class_names) if class_names else None

    records: list[dict[str, Any]] = []
    for sample in manifest:
        image_path = str(sample.get("local_path", ""))
        boxes: list[list[float]] = []
        labels: list[str] = []

        for ann in sample.get("annotations", []):
            class_name = str(ann.get("category_name", "")).strip()
            if allowed is not None and class_name not in allowed:
                continue
            if not class_name:
                continue

            centered = ann.get("bbox_xywh_centered", {})
            if not isinstance(centered, dict):
                continue

            box = _centered_dict_to_cxcywh(centered)
            boxes.append(box)
            labels.append(class_name)

        records.append(
            {
                "image_path": image_path,
                "boxes": boxes,
                "labels": labels,
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, indent=2)


def export_manifest_to_coco(
    split: str,
    output_path: str | Path,
    class_names: list[str] | None = None,
) -> None:
    """Export the default split manifest to COCO-style JSON."""
    manifest = _load_manifest_from_split(split)
    _export_manifest_records_to_coco(manifest, output_path, class_names)


def export_manifest_to_eval_json(
    split: str,
    output_path: str | Path,
    class_names: list[str] | None = None,
) -> None:
    """Export the default split manifest to eval-format JSON."""
    manifest = _load_manifest_from_split(split)
    _export_manifest_records_to_eval_json(manifest, output_path, class_names)


def export_manifest_file_to_coco(
    manifest_path: str | Path,
    output_path: str | Path,
    class_names: list[str] | None = None,
) -> None:
    """Export an explicit manifest file to COCO-style JSON."""
    manifest = _load_manifest_from_path(manifest_path)
    _export_manifest_records_to_coco(manifest, output_path, class_names)


def export_manifest_file_to_eval_json(
    manifest_path: str | Path,
    output_path: str | Path,
    class_names: list[str] | None = None,
) -> None:
    """Export an explicit manifest file to eval-format JSON."""
    manifest = _load_manifest_from_path(manifest_path)
    _export_manifest_records_to_eval_json(manifest, output_path, class_names)
