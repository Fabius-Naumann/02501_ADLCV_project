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
        threshold: float = 0.3,
        multiclass_prompt: bool = True,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.use_prompt_aliases = use_prompt_aliases
        self.threshold = threshold
        self.multiclass_prompt = multiclass_prompt

    def _to_prompt(self, category_name: str) -> str:
        if not self.use_prompt_aliases:
            return category_name.replace("_", " ")
        return PROMPT_ALIASES.get(category_name, category_name.replace("_", " "))

    def _post_process(
        self,
        outputs: Any,
        input_ids: torch.Tensor,
        image_pil: Image.Image,
        threshold: float,
    ) -> dict[str, torch.Tensor]:
        target_sizes = [image_pil.size[::-1]]

        try:
            return self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids,
                box_threshold=threshold,
                text_threshold=threshold,
                target_sizes=target_sizes,
            )[0]
        except TypeError:
            return self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids=input_ids,
                threshold=threshold,
                target_sizes=target_sizes,
            )[0]

    @staticmethod
    def _normalize_prompt_label(label: Any) -> str:
        text = label if isinstance(label, str) else str(label)
        return " ".join(text.replace("_", " ").replace(".", " ").lower().split())

    def _map_grounding_label(
        self,
        raw_label: Any,
        prompt_to_original: dict[str, str],
        fallback_label: str | None = None,
    ) -> str | None:
        normalized_raw = self._normalize_prompt_label(raw_label)
        if not normalized_raw and fallback_label is not None:
            return fallback_label

        normalized_prompt_to_original = {
            self._normalize_prompt_label(prompt): original for prompt, original in prompt_to_original.items()
        }

        if normalized_raw in normalized_prompt_to_original:
            return normalized_prompt_to_original[normalized_raw]

        for normalized_prompt, original in normalized_prompt_to_original.items():
            if normalized_prompt in normalized_raw or normalized_raw in normalized_prompt:
                return original

        return fallback_label

    def _predict_single_prompt(
        self,
        image_pil: Image.Image,
        text_prompt: str,
        threshold: float,
    ) -> dict[str, Any]:
        inputs = self.processor(
            images=image_pil,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return self._post_process(
            outputs=outputs,
            input_ids=inputs.input_ids,
            image_pil=image_pil,
            threshold=threshold,
        )

    def predict(
        self,
        image_tensor: Tensor,
        category_names: list[str],
        threshold: float | None = None,
    ) -> dict[str, Tensor | list[str]]:
        all_boxes: list[Tensor] = []
        all_scores: list[Tensor] = []
        all_labels: list[str] = []

        resolved_threshold = self.threshold if threshold is None else threshold

        image_tensor_cpu = image_tensor.detach().cpu().clamp(0, 1)
        image_pil = Image.fromarray((image_tensor_cpu.permute(1, 2, 0).numpy() * 255).astype("uint8"))

        cleaned_category_names = list(dict.fromkeys([name.strip() for name in category_names if name.strip()]))
        if not cleaned_category_names:
            return {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "scores": torch.empty((0,), dtype=torch.float32),
                "labels": [],
            }

        prompt_categories = [self._to_prompt(category_name) for category_name in cleaned_category_names]
        prompt_to_original = dict(zip(prompt_categories, cleaned_category_names, strict=True))

        if self.multiclass_prompt:
            text_prompt = " . ".join(prompt_categories) + "."
            result = self._predict_single_prompt(
                image_pil=image_pil,
                text_prompt=text_prompt,
                threshold=resolved_threshold,
            )

            boxes = result["boxes"].detach().cpu().to(torch.float32)
            scores = result["scores"].detach().cpu().to(torch.float32)
            raw_labels = result.get("labels", result.get("text_labels", []))
            if raw_labels is None or len(raw_labels) != len(boxes):
                raw_labels = [""] * len(boxes)

            for box, score, raw_label in zip(boxes, scores, raw_labels, strict=True):
                mapped_label = self._map_grounding_label(raw_label, prompt_to_original)
                if mapped_label is None:
                    continue
                all_boxes.append(box)
                all_scores.append(score)
                all_labels.append(mapped_label)
        else:
            # Legacy ablation mode: query one class at a time.
            for category_name, prompt_category in zip(cleaned_category_names, prompt_categories, strict=True):
                result = self._predict_single_prompt(
                    image_pil=image_pil,
                    text_prompt=prompt_category + ".",
                    threshold=resolved_threshold,
                )
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
        device: str | None = None,
        use_prompt_aliases: bool = True,
    ) -> None:
        self.model = YOLOWorld(model_id)
        self.imgsz = imgsz
        self.conf = conf
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
    _TASK2_OBJECT_DETECTION_BOUNDED_BOXES = (
        "You are a helpful assistant to detect objects in images in a few-shot setting. "
        "You will be provided with support example images where the relevant objects are "
        "highlighted with red bounding boxes. "
        "Use these examples to understand what the target object looks like, then apply this knowledge "
        "to detect similar objects in the query image. "
        "When asked to detect elements based on a description, return ONLY valid JSON. "
        'Return a JSON array in this form: [{"bbox_2d": [xmin, ymin, xmax, ymax], "label": "class_name", '
        '"score": 0.0}]. '
        "Coordinates must be integers scaled to a fixed 1000x1000 reference frame, where each value is in [0, 1000]. "
        "Do not output absolute image pixel coordinates. Enforce xmin < xmax and ymin < ymax. "
        "If no object is present, return []. "
        "Do not include markdown, comments, or any extra text."
    )
    _TASK2_OBJECT_DETECTION_MARKED = (
        "You are a helpful assistant to detect objects in images in a few-shot setting. "
        "You will be provided with support example images where the relevant objects are "
        "highlighted with red mark on them. "
        "Use these examples to understand what the target object looks like, then apply this knowledge "
        "to detect similar objects in the query image. "
        "When asked to detect elements based on a description, return ONLY valid JSON. "
        'Return a JSON array in this form: [{"bbox_2d": [xmin, ymin, xmax, ymax], "label": "class_name", '
        '"score": 0.0}]. '
        "Coordinates must be integers scaled to a fixed 1000x1000 reference frame, where each value is in [0, 1000]. "
        "Do not output absolute image pixel coordinates. Enforce xmin < xmax and ymin < ymax. "
        "If no object is present, return []. "
        "Do not include markdown, comments, or any extra text."
    )
    _TASK2_OBJECT_DETECTION_CROPPED = (
        "You are a helpful assistant to detect objects in images in a few-shot setting. "
        "You will be provided with support example images, which are cropped to focus on the relevant objects. "
        "Use these examples to understand what the target object looks like, then apply this knowledge "
        "to detect similar objects in the query image. "
        "When asked to detect elements based on a description, return ONLY valid JSON. "
        'Return a JSON array in this form: [{"bbox_2d": [xmin, ymin, xmax, ymax], "label": "class_name", '
        '"score": 0.0}]. '
        "Coordinates must be integers scaled to a fixed 1000x1000 reference frame, where each value is in [0, 1000]. "
        "Do not output absolute image pixel coordinates. Enforce xmin < xmax and ymin < ymax. "
        "If no object is present, return []. "
        "Do not include markdown, comments, or any extra text."
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

    def _build_task2_support_conditioned_prompt(self, category_name: str, max_detections: int) -> str:
        """Build a direct few-shot detection prompt using separate support and query images."""
        return (
            "You are provided with two images. The first image is a support panel containing examples of "
            f"the target object class '{category_name}'. The second image is the query image. "
            f"Detect objects of class '{category_name}' only in the query image. "
            f"Set each detection label to '{category_name}'. "
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
        image_pil: Image.Image | None = None,
        prompt: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        image_pils: list[Image.Image] | None = None,
    ) -> str:
        resolved_system_prompt = system_prompt or self._SYSTEM_PROMPT_OBJECT_DETECTION
        used_chat_template = False
        images = image_pils if image_pils is not None else ([image_pil] if image_pil is not None else [])
        if not images:
            raise ValueError("At least one image must be provided.")
        try:
            messages = [
                {"role": "system", "content": resolved_system_prompt},
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in images],
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(text=[text_prompt], images=images, return_tensors="pt")
            used_chat_template = True
        except (AttributeError, TypeError, ValueError):
            fallback_prompt = f"{resolved_system_prompt}\n\n{prompt}"
            inputs = self.processor(images=images, text=fallback_prompt, return_tensors="pt")

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
        system_prompt: str | None = None,
    ) -> dict[str, Tensor | list[str] | list[dict[str, Any]]]:
        """Run prompt-based localization for a list of categories.

        Args:
            image_tensor: C x H x W tensor from Task1DetectionDataset.
            category_names: List of labels to query.
            max_detections_per_category: Maximum detections returned per category.
            max_new_tokens: Generation length limit.
            temperature: Decoding temperature.
            return_debug_outputs: Include per-category input/raw/parsed/final debug payload.
            system_prompt: Optional override for the detection system prompt.

        Returns:
            Dictionary with keys: boxes (cxcywh), scores, labels.
            If ``return_debug_outputs=True``, includes ``debug_entries``.
        """
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
                system_prompt=system_prompt,
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
        system_prompt: str | None = None,
    ) -> str:
        raw_description = self._generate_text(
            image_pil=support_image,
            prompt=self._build_description_prompt(category_name=category_name),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt or self._SYSTEM_PROMPT_VISUAL_DESCRIPTION,
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
        system_prompt: str | None = None,
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
            system_prompt=system_prompt,
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

    def predict_with_support_query_panel(
        self,
        support_query_panel: Image.Image,
        category_name: str,
        query_image_width: int,
        query_image_height: int,
        max_detections: int = 1,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        return_debug_outputs: bool = False,
        system_prompt: str | None = None,
    ) -> dict[str, Tensor | list[str] | list[dict[str, Any]]]:
        """Detect query objects directly from a support-query panel without text-from-vision."""
        normalized_category_name = category_name.strip()
        if not normalized_category_name:
            raise ValueError("category_name must be non-empty.")

        prompt = self._build_task2_support_conditioned_prompt(
            category_name=normalized_category_name,
            max_detections=max_detections,
        )
        resolved_system_prompt = (
            system_prompt
            or getattr(self, "_SYSTEM_PROMPT_OBJECT_DETECTION", None)
            or getattr(self, "_TASK2_OBJECT_DETECTION_BOUNDED_BOXES", None)
            or getattr(self, "_TASK2_OBJECT_DETECTION_CROPPED", None)
            or getattr(self, "_TASK2_OBJECT_DETECTION_MARKED", "")
        )
        generated_text = self._generate_text(
            image_pil=support_query_panel,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=resolved_system_prompt,
        )
        parsed_detections, parse_metadata = self._parse_generated_output(generated_text)
        boxes, scores, labels = self._extract_category_detections(
            detections=parsed_detections,
            category_name=normalized_category_name,
            image_width=query_image_width,
            image_height=query_image_height,
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
                    "category_name": normalized_category_name,
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

    def predict_with_support_panel(
        self,
        query_image_tensor: Tensor,
        support_panel_pil: Image.Image,
        category_name: str,
        query_image_width: int,
        query_image_height: int,
        max_detections: int = 1,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        return_debug_outputs: bool = False,
        system_prompt: str | None = None,
    ) -> dict[str, Tensor | list[str] | list[dict[str, Any]]]:
        """Detect query objects using a support panel and separate query image.

        Args:
            query_image_tensor: Query image tensor (C x H x W) in [0, 1].
            support_panel_pil: PIL.Image with all support examples combined into one panel.
            category_name: Target object category name.
            query_image_width: Width of query image in pixels.
            query_image_height: Height of query image in pixels.
            max_detections: Maximum detections per category.
            max_new_tokens: Maximum tokens to generate.
            temperature: Generation temperature.
            return_debug_outputs: Whether to return debug information.
            system_prompt: Optional system prompt override.

        Returns:
            Dictionary with boxes (cxcywh), scores, and labels.
        """
        normalized_category_name = category_name.strip()
        if not normalized_category_name:
            raise ValueError("category_name must be non-empty.")

        # Convert query tensor to PIL
        query_image_np = query_image_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        query_image_pil = Image.fromarray((query_image_np * 255).astype("uint8"))

        # Build the prompt
        prompt = self._build_task2_support_conditioned_prompt(
            category_name=normalized_category_name,
            max_detections=max_detections,
        )

        generated_text = self._generate_text(
            image_pils=[support_panel_pil, query_image_pil],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt or self._TASK2_OBJECT_DETECTION_BOUNDED_BOXES,
        )

        # Parse output
        parsed_detections, parse_metadata = self._parse_generated_output(generated_text)

        boxes, scores, labels = self._extract_category_detections(
            detections=parsed_detections,
            category_name=normalized_category_name,
            image_width=query_image_width,
            image_height=query_image_height,
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
                    "category_name": normalized_category_name,
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

    def predict_with_support_images(
        self,
        query_image_tensor: Tensor,
        support_images_pil: list[Image.Image],
        category_name: str,
        query_image_width: int,
        query_image_height: int,
        max_detections: int = 1,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        return_debug_outputs: bool = False,
        system_prompt: str | None = None,
    ) -> dict[str, Tensor | list[str] | list[dict[str, Any]]]:
        """Detect query objects using separate support and query images.

        Args:
            query_image_tensor: Query image tensor (C x H x W) in [0, 1].
            support_images_pil: List of support PIL.Images with highlighted objects.
            category_name: Target object category name.
            query_image_width: Width of query image in pixels.
            query_image_height: Height of query image in pixels.
            max_detections: Maximum detections per category.
            max_new_tokens: Maximum tokens to generate.
            temperature: Generation temperature.
            return_debug_outputs: Whether to return debug information.
            system_prompt: Optional system prompt override.

        Returns:
            Dictionary with boxes (cxcywh), scores, and labels.
        """
        normalized_category_name = category_name.strip()
        if not normalized_category_name:
            raise ValueError("category_name must be non-empty.")

        # Convert query tensor to PIL
        query_image_np = query_image_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
        query_image_pil = Image.fromarray((query_image_np * 255).astype("uint8"))

        # Build the prompt
        prompt = self._build_task2_support_conditioned_prompt(
            category_name=normalized_category_name,
            max_detections=max_detections,
        )

        # For now, concatenate support images vertically and query image horizontally
        # Create a combined image with all support examples above and query below
        if support_images_pil:
            # Stack support images horizontally
            support_width = sum(img.width for img in support_images_pil) + 8 * (len(support_images_pil) - 1)
            support_height = max(img.height for img in support_images_pil)
            support_canvas = Image.new("RGB", (support_width, support_height), color=(255, 255, 255))
            x_offset = 0
            for support_img in support_images_pil:
                support_canvas.paste(support_img, (x_offset, 0))
                x_offset += support_img.width + 8

            # Resize query image to match support height
            query_resized = query_image_pil.copy()
            if query_resized.height != support_height:
                new_width = max(1, round(query_resized.width * (support_height / query_resized.height)))
                query_resized = query_resized.resize((new_width, support_height), Image.Resampling.BILINEAR)

            # Combine support and query side by side
            combined_width = support_canvas.width + query_resized.width + 8
            combined_height = support_height
            combined_canvas = Image.new("RGB", (combined_width, combined_height), color=(255, 255, 255))
            combined_canvas.paste(support_canvas, (0, 0))
            combined_canvas.paste(query_resized, (support_canvas.width + 8, 0))
            input_image_pil = combined_canvas
        else:
            input_image_pil = query_image_pil

        # Generate detections
        generated_text = self._generate_text(
            image_pil=input_image_pil,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt or self._TASK2_OBJECT_DETECTION_BOUNDED_BOXES,
        )

        # Parse output
        parsed_detections, parse_metadata = self._parse_generated_output(generated_text)
        boxes, scores, labels = self._extract_category_detections(
            detections=parsed_detections,
            category_name=normalized_category_name,
            image_width=query_image_width,
            image_height=query_image_height,
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
                    "category_name": normalized_category_name,
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
