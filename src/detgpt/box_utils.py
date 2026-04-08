from __future__ import annotations

import torch
from torch import Tensor


def xywh_to_cxcywh(box: list[float]) -> list[float]:
    """Convert [x_min, y_min, width, height] to [center_x, center_y, width, height].

    Args:
        box: Bounding box in xywh format.

    Returns:
        Bounding box in cxcywh format.
    """
    x_min, y_min, width, height = box
    center_x = x_min + width / 2.0
    center_y = y_min + height / 2.0
    return [center_x, center_y, width, height]


def xywh_to_cxcywh_dict(box: list[float]) -> dict[str, float]:
    """Convert [x_min, y_min, width, height] to a named center-format dictionary.

    Args:
        box: Bounding box in xywh format.

    Returns:
        Dictionary with keys x_center, y_center, width, height.
    """
    center_x, center_y, width, height = xywh_to_cxcywh(box)
    return {
        "x_center": center_x,
        "y_center": center_y,
        "width": width,
        "height": height,
    }


def cxcywh_to_xyxy(box: list[float]) -> list[float]:
    """Convert [center_x, center_y, width, height] to [x1, y1, x2, y2].

    Args:
        box: Bounding box in cxcywh format.

    Returns:
        Bounding box in xyxy format.
    """
    center_x, center_y, width, height = box
    x1 = center_x - width / 2.0
    y1 = center_y - height / 2.0
    x2 = center_x + width / 2.0
    y2 = center_y + height / 2.0
    return [x1, y1, x2, y2]


def xyxy_to_cxcywh(box: list[float]) -> list[float]:
    """Convert [x1, y1, x2, y2] to [center_x, center_y, width, height].

    Args:
        box: Bounding box in xyxy format.

    Returns:
        Bounding box in cxcywh format.
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2.0
    center_y = y1 + height / 2.0
    return [center_x, center_y, width, height]


def cxcywh_tensor_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert N x 4 boxes from cxcywh to xyxy format.

    Args:
        boxes: Tensor with shape [N, 4] in cxcywh format.

    Returns:
        Tensor with shape [N, 4] in xyxy format.
    """
    if boxes.numel() == 0:
        return boxes

    center_x = boxes[:, 0]
    center_y = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]

    x1 = center_x - width / 2.0
    y1 = center_y - height / 2.0
    x2 = center_x + width / 2.0
    y2 = center_y + height / 2.0
    return torch.stack((x1, y1, x2, y2), dim=1)


def compute_iou_cxcywh(box1: list[float], box2: list[float]) -> float:
    """Compute IoU for two boxes represented in cxcywh format.

    Args:
        box1: First bounding box in cxcywh format.
        box2: Second bounding box in cxcywh format.

    Returns:
        Intersection-over-union value.
    """
    x1_a, y1_a, x2_a, y2_a = cxcywh_to_xyxy(box1)
    x1_b, y1_b, x2_b, y2_b = cxcywh_to_xyxy(box2)

    inter_x1 = max(x1_a, x1_b)
    inter_y1 = max(y1_a, y1_b)
    inter_x2 = min(x2_a, x2_b)
    inter_y2 = min(y2_a, y2_b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
    area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)

    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union
