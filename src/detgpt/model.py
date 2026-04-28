from __future__ import annotations

import json
import os
import re
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoModelForImageTextToText, AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLOWorld

from detgpt.box_utils import xyxy_to_cxcywh

PROMPT_ALIASES = {
    "cincture": "decorative cincture belt worn around the waist",
    "yoke_(animal_equipment)": "wooden animal yoke equipment for oxen or horses",
    "knocker_(on_a_door)": "metal door knocker mounted on a door",
    "poker_(fire_stirring_tool)": "long metal fireplace poker tool",
    "pew_(church_bench)": "wooden church pew bench",
    "mail_slot": "rectangular mail slot on a door",
    "cufflink": "small metal cufflink on shirt cuff",
    "oil_lamp": "antique oil lamp with glass chimney",
    "gravy_boat": "ceramic gravy boat sauce boat with spout and handle",
    "quiche": "round savory quiche tart",
}


class Model(nn.Module):
    """Minimal dummy model scaffold."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class GroundingDINOHandler:
    """Wrapper for Grounding DINO zero-shot object detection."""

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-tiny",
        use_prompt_aliases: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.use_prompt_aliases = use_prompt_aliases

    def _to_prompt(self, category_name: str) -> str:
        if not self.use_prompt_aliases:
            return category_name.replace("_", " ")
        return PROMPT_ALIASES.get(category_name, category_name.replace("_", " "))

    def predict(
        self,
        image_tensor: Tensor,
        category_names: list[str],
        threshold: float = 0.3,
    ) -> dict[str, Tensor | list[str]]:
        all_boxes = []
        all_scores = []
        all_labels = []

        image_tensor_cpu = image_tensor.detach().cpu().clamp(0, 1)
        image_pil = Image.fromarray((image_tensor_cpu.permute(1, 2, 0).numpy() * 255).astype("uint8"))

        cleaned_category_names = list(dict.fromkeys([name.strip() for name in category_names if name.strip()]))

        for category_name in cleaned_category_names:
            text_prompt = self._to_prompt(category_name) + "."

            inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            result = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=threshold,
                text_threshold=threshold,
                target_sizes=[image_pil.size[::-1]],
            )[0]

            boxes = result["boxes"].detach().cpu().to(torch.float32)
            scores = result["scores"].detach().cpu().to(torch.float32)

            for box, score in zip(boxes, scores, strict=True):
                all_boxes.append(box)
                all_scores.append(score)
                all_labels.append(category_name)

        if all_boxes:
            boxes_tensor = torch.stack(all_boxes).to(torch.float32)
            scores_tensor = torch.stack(all_scores).to(torch.float32)
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
            scores_tensor = torch.empty((0,), dtype=torch.float32)

        return {
            "boxes": boxes_tensor,
            "scores": scores_tensor,
            "labels": all_labels,
        }


class YOLOWorldHandler:
    """Wrapper for YOLO-World zero-shot object detection."""

    def __init__(
        self,
        model_id: str = "yolov8s-world.pt",
        imgsz: int = 640,
        conf: float = 0.05,
        device: str = "cpu",
        use_prompt_aliases: bool = True,
    ) -> None:
        self.model = YOLOWorld(model_id)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.use_prompt_aliases = use_prompt_aliases
        self.model.to(self.device)

    def _to_prompt(self, category_name: str) -> str:
        if not self.use_prompt_aliases:
            return category_name.replace("_", " ")
        return PROMPT_ALIASES.get(category_name, category_name.replace("_", " "))

    def predict(self, image: Tensor, query_categories: list[str]) -> dict[str, Tensor | list[str]]:
        if not query_categories:
            return {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "scores": torch.empty((0,), dtype=torch.float32),
                "labels": [],
            }

        query_categories = list(dict.fromkeys([name.strip() for name in query_categories if name.strip()]))

        prompt_categories = [self._to_prompt(category_name) for category_name in query_categories]
        prompt_to_original = dict(zip(prompt_categories, query_categories, strict=True))

        self.model.set_classes(prompt_categories)

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
        class_indices = [int(value) for value in result.boxes.cls.detach().cpu().tolist()]

        labels = []
        for index in class_indices:
            prompt_label = prompt_categories[index]
            labels.append(prompt_to_original[prompt_label])

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }


class QwenVLMHandler:
    """Wrapper for prompt-based object localization with Qwen VLMs."""

    _REFERENCE_COORD_MAX = 1000.0
    _SYSTEM_PROMPT_OBJECT_DETECTION = (
        "You are a helpful assistant to detect objects in images. "
        "When asked to detect elements based on a description, return ONLY valid JSON. "
        'Return a JSON array in this form: [{"bbox_2d": [xmin, ymin, xmax, ymax], "label": "class_name", '
        '"score": 0.0}]. '
        "Coordinates must be integers scaled to a fixed 1000x1000 reference frame, where each value is in [0, 1000]. "
        "Do not output absolute image pixel coordinates. Enforce xmin < xmax and ymin < ymax. "
        "If no object is present, return []. "
        "Do not include markdown, comments, or any extra text."
    )
    _SYSTEM_PROMPT_VISUAL_DESCRIPTION = (
        "You are a helpful assistant for visual attribute extraction from support examples. "
        "You will be shown one or more support images where the relevant objects are highlighted with red boxes. "
        "Describe the boxed objects using concise, visually distinguishing traits only. "
        "Focus on appearance, shape, parts, texture and material that help someone find the same or similar "
        "object in another image. Do not describe the whole image. Do not mention coordinates or the red box."
    )

    def __init__(self, model_id: str = "Qwen/Qwen3.5-2B") -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=hf_token,
        )

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            token=hf_token,
        ).to(self.device)

        self.model.eval()

    def _build_prompt(self, category_name: str, max_detections: int) -> str:
        visual_prompt = PROMPT_ALIASES.get(category_name, category_name.replace("_", " "))
        return (
            f"Detect objects of class '{category_name}' in this image. "
            f"The visual description is: {visual_prompt}. "
            f"Set each detection 'label' field to '{category_name}'. "
            f"Return at most {max_detections} detection(s). "
            "Prefer high precision over high recall."
        )

    def _build_description_prompt(self, category_name: str) -> str:
        return (
            f"The red box shows an example of '{category_name}'. "
            "Write a short description of the boxed object using only visually distinguishing traits. "
            "Keep it to one or two sentences. Mention appearance, shape, material, parts, and local context if useful. "
            "Do not mention the class name unless it is necessary to avoid ambiguity."
        )

    def _build_description_detection_prompt(
        self,
        description: str,
        output_label: str,
        max_detections: int,
    ) -> str:
        return (
            "Detect objects in this image that match the following support-derived description: "
            f"{description.strip()} "
            f"Set each detection 'label' field to '{output_label}'. "
            f"Return at most {max_detections} detection(s). "
            "Prefer high precision over high recall."
        )

    @staticmethod
    def _extract_json_blob(generated_text: str) -> Any | None:
        decoder = json.JSONDecoder()
        for start_index, char in enumerate(generated_text):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(generated_text[start_index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, (dict, list)):
                return parsed
        return None

    @staticmethod
    def _normalize_json_detections(parsed_json: Any) -> list[dict[str, list[float] | float | str]]:
        if isinstance(parsed_json, dict):
            detections_value = parsed_json.get("detections", [])
            raw_detections = detections_value if isinstance(detections_value, list) else []
        elif isinstance(parsed_json, list):
            raw_detections = parsed_json
        else:
            raw_detections = []

        normalized: list[dict[str, list[float] | float | str]] = []
        for detection in raw_detections:
            if isinstance(detection, dict):
                box_value = detection.get("bbox_2d", detection.get("bbox_xyxy"))
                label_value = detection.get("label", "")
                score_value = detection.get("score", 1.0)
            elif isinstance(detection, list) and len(detection) == 4:
                box_value = detection
                label_value = ""
                score_value = 1.0
            else:
                continue

            if isinstance(box_value, tuple):
                box_value = list(box_value)
            if not isinstance(box_value, list) or len(box_value) != 4:
                continue

            normalized.append(
                {
                    "bbox_2d": box_value,
                    "bbox_xyxy": box_value,
                    "label": str(label_value),
                    "score": score_value,
                }
            )

        return normalized

    @staticmethod
    def _extract_coordinate_pair_detections(generated_text: str) -> list[dict[str, list[float] | float]]:
        pattern = re.compile(
            r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*,\s*"
            r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)"
        )
        matches = pattern.findall(generated_text)
        detections: list[dict[str, list[float] | float]] = []
        for x1, y1, x2, y2 in matches:
            detections.append(
                {
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "score": 1.0,
                }
            )
        return detections

    def _parse_generated_output(
        self,
        generated_text: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        json_blob = self._extract_json_blob(generated_text)

        if isinstance(json_blob, list):
            json_detections = self._normalize_json_detections(json_blob)
            if json_detections or len(json_blob) == 0:
                return json_detections, {
                    "parser": "json",
                    "parsed_output": {"detections": json_detections},
                }

        if isinstance(json_blob, dict):
            detections_value = json_blob.get("detections")
            if isinstance(detections_value, list):
                json_detections = self._normalize_json_detections(json_blob)
                if json_detections or len(detections_value) == 0:
                    return json_detections, {
                        "parser": "json",
                        "parsed_output": {"detections": json_detections},
                    }

        coord_detections = self._extract_coordinate_pair_detections(generated_text)
        if coord_detections:
            return coord_detections, {
                "parser": "coordinate_pairs",
                "parsed_output": {"detections": coord_detections},
            }

        return [], {
            "parser": "none",
            "parsed_output": {"detections": []},
        }

    def _generate_text(
        self,
        image_pil: Image.Image,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        system_prompt: str | None = None,
    ) -> str:
        resolved_system_prompt = system_prompt or self._SYSTEM_PROMPT_OBJECT_DETECTION
        used_chat_template = False

        try:
            messages = [
                {"role": "system", "content": resolved_system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_pil},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(text=[text_prompt], images=[image_pil], return_tensors="pt")
            used_chat_template = True
        except (AttributeError, TypeError, ValueError):
            fallback_prompt = f"{resolved_system_prompt}\n\n{prompt}"
            inputs = self.processor(images=image_pil, text=fallback_prompt, return_tensors="pt")

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        if used_chat_template and "input_ids" in inputs:
            prompt_length = inputs["input_ids"].shape[-1]
            generated_ids = generated_ids[:, prompt_length:]

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return generated_text.strip()

    @staticmethod
    def _normalize_description(description: str) -> str:
        description_text = " ".join(description.strip().split())
        if not description_text:
            raise ValueError("Generated description is empty.")
        return description_text

    def _extract_category_detections(
        self,
        detections: list[dict[str, Any]],
        category_name: str,
        image_width: int,
        image_height: int,
        max_detections_per_category: int,
    ) -> tuple[list[list[float]], list[float], list[str]]:
        boxes: list[list[float]] = []
        scores: list[float] = []
        labels: list[str] = []

        reference_max = self._REFERENCE_COORD_MAX
        width_scale = float(max(image_width - 1, 0)) / reference_max
        height_scale = float(max(image_height - 1, 0)) / reference_max

        for detection in detections[:max_detections_per_category]:
            if not isinstance(detection, dict):
                continue

            box = detection.get("bbox_xyxy")
            score = detection.get("score", 0.0)

            if not isinstance(box, list) or len(box) != 4:
                continue

            try:
                x1, y1, x2, y2 = [float(value) for value in box]
                score_value = float(score)
            except (TypeError, ValueError):
                continue

            x1 = min(max(x1, 0.0), reference_max)
            y1 = min(max(y1, 0.0), reference_max)
            x2 = min(max(x2, 0.0), reference_max)
            y2 = min(max(y2, 0.0), reference_max)

            if x2 <= x1 or y2 <= y1:
                continue

            x1 = x1 * width_scale
            y1 = y1 * height_scale
            x2 = x2 * width_scale
            y2 = y2 * height_scale

            boxes.append(xyxy_to_cxcywh([x1, y1, x2, y2]))
            scores.append(score_value)
            labels.append(category_name)

        return boxes, scores, labels

    def predict(
        self,
        image_tensor: Tensor,
        category_names: list[str],
        max_detections_per_category: int = 1,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        return_debug_outputs: bool = False,
    ) -> dict[str, Tensor | list[str] | list[dict[str, Any]]]:
        image_tensor_cpu = image_tensor.detach().cpu().clamp(0, 1)
        image_pil = Image.fromarray((image_tensor_cpu.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        image_width, image_height = image_pil.size

        unique_category_names = list(dict.fromkeys([name.strip() for name in category_names if name.strip()]))

        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []
        debug_entries: list[dict[str, Any]] = []

        for category_name in unique_category_names:
            prompt = self._build_prompt(
                category_name=category_name,
                max_detections=max_detections_per_category,
            )
            generated_text = self._generate_text(
                image_pil=image_pil,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            parsed_detections, parse_metadata = self._parse_generated_output(generated_text)

            category_boxes, category_scores, category_labels = self._extract_category_detections(
                detections=parsed_detections,
                category_name=category_name,
                image_width=image_width,
                image_height=image_height,
                max_detections_per_category=max_detections_per_category,
            )

            if return_debug_outputs:
                debug_entries.append(
                    {
                        "category_name": category_name,
                        "input_prompt": prompt,
                        "raw_output_text": generated_text,
                        "parsed_output": parse_metadata,
                        "final_output": {
                            "boxes_cxcywh": category_boxes,
                            "scores": category_scores,
                            "labels": category_labels,
                        },
                    }
                )

            all_boxes.extend(category_boxes)
            all_scores.extend(category_scores)
            all_labels.extend(category_labels)

        if all_boxes:
            boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
            scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            scores_tensor = torch.zeros((0,), dtype=torch.float32)

        outputs: dict[str, Tensor | list[str] | list[dict[str, Any]]] = {
            "boxes": boxes_tensor,
            "scores": scores_tensor,
            "labels": all_labels,
        }

        if return_debug_outputs:
            outputs["debug_entries"] = debug_entries

        return outputs

    def generate_support_description(
        self,
        support_image: Image.Image,
        category_name: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        raw_description = self._generate_text(
            image_pil=support_image,
            prompt=self._build_description_prompt(category_name=category_name),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=self._SYSTEM_PROMPT_VISUAL_DESCRIPTION,
        )
        return self._normalize_description(raw_description)

    def predict_from_description(
        self,
        image_tensor: Tensor,
        description: str,
        output_label: str,
        max_detections: int = 1,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        return_debug_outputs: bool = False,
    ) -> dict[str, Tensor | list[str] | list[dict[str, Any]]]:
        normalized_description = self._normalize_description(description)
        image_tensor_cpu = image_tensor.detach().cpu().clamp(0, 1)
        image_pil = Image.fromarray((image_tensor_cpu.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        image_width, image_height = image_pil.size

        prompt = self._build_description_detection_prompt(
            description=normalized_description,
            output_label=output_label,
            max_detections=max_detections,
        )
        generated_text = self._generate_text(
            image_pil=image_pil,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        parsed_detections, parse_metadata = self._parse_generated_output(generated_text)

        boxes, scores, labels = self._extract_category_detections(
            detections=parsed_detections,
            category_name=output_label,
            image_width=image_width,
            image_height=image_height,
            max_detections_per_category=max_detections,
        )

        outputs: dict[str, Tensor | list[str] | list[dict[str, Any]]] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros((0,), dtype=torch.float32),
            "labels": labels,
        }

        if return_debug_outputs:
            outputs["debug_entries"] = [
                {
                    "description": normalized_description,
                    "input_prompt": prompt,
                    "raw_output_text": generated_text,
                    "parsed_output": parse_metadata,
                    "final_output": {
                        "boxes_cxcywh": boxes,
                        "scores": scores,
                        "labels": labels,
                    },
                }
            ]

        return outputs


if __name__ == "__main__":
    model = Model()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
