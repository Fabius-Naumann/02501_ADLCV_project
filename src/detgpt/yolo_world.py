from __future__ import annotations

import torch
from torch import Tensor
from ultralytics import YOLOWorld


class YOLOWorldHandler:
    def __init__(
        self,
        model_id: str = "yolov8s-world.pt",
        imgsz: int = 640,
        conf: float = 0.05,
        device: str = "cpu",
    ) -> None:
        self.model = YOLOWorld(model_id)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device

        # Force the whole YOLO-World pipeline onto CPU to avoid the
        # CLIP text-encoder device mismatch in set_classes(...)
        self.model.to(self.device)

    def predict(self, image: Tensor, query_categories: list[str]) -> dict[str, Tensor | list[str]]:
        if not query_categories:
            return {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "scores": torch.empty((0,), dtype=torch.float32),
                "labels": [],
            }

        self.model.set_classes(query_categories)

        # Ultralytics expects HWC numpy input
        image_np = image.permute(1, 2, 0).mul(255).clamp(0, 255).byte().cpu().numpy()

        results = self.model.predict(
            source=image_np,
            imgsz=self.imgsz,
            conf=self.conf,
            verbose=False,
            device=self.device,
        )

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "scores": torch.empty((0,), dtype=torch.float32),
                "labels": [],
            }

        boxes = result.boxes.xyxy.detach().cpu().to(torch.float32)
        scores = result.boxes.conf.detach().cpu().to(torch.float32)
        class_indices = [int(x) for x in result.boxes.cls.detach().cpu().tolist()]
        labels = [query_categories[i] for i in class_indices]

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }
