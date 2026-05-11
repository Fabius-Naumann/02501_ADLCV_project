from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch import Tensor, nn
from transformers import AutoModelForImageTextToText, AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import YOLOWorld

from detgpt.box_utils import xyxy_to_cxcywh
from detgpt.device import DeviceSpec, resolve_torch_device


class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class GroundingDINOHandler:
    """Wrapper for Grounding DINO zero-shot object detection."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: DeviceSpec = None):
        """Initialize model and processor.

        Args:
            model_id: Hugging Face model identifier.
            device: PyTorch device. Use ``None`` or ``"auto"`` for CUDA, MPS, then CPU.
        """
        self.device = resolve_torch_device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def predict(
        self,
        image_tensor: Tensor,
        category_names: list[str],
        threshold: float = 0.3,
    ) -> dict[str, Tensor | list[str]]:
        """
        Run inference on a single image tensor.

        Args:
            image_tensor: C x H x W tensor from Task1DetectionDataset.
            category_names: List of labels to find (e.g., ["cat", "dog"]).
            threshold: Confidence threshold for detections.

        Returns:
            Dictionary with:
                - boxes: Tensor[N, 4] in xyxy pixel coordinates
                - scores: Tensor[N]
                - labels: list[str] of length N
        """
        cleaned_category_names = [name.strip() for name in category_names if name.strip()]
        unique_category_names = list(dict.fromkeys(cleaned_category_names))
        text_prompt = ". ".join(unique_category_names) + "."

        image_tensor_cpu = image_tensor.detach().cpu().clamp(0, 1)
        image_pil = Image.fromarray((image_tensor_cpu.permute(1, 2, 0).numpy() * 255).astype("uint8"))

        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image_pil.size[::-1]],
        )[0]

        boxes = result["boxes"]
        scores = result["scores"]

        raw_labels = result["text_labels"] if "text_labels" in result else result["labels"]
        labels = [str(label) for label in raw_labels]

        count = min(len(boxes), len(scores), len(labels))
        boxes = boxes[:count]
        scores = scores[:count]
        labels = labels[:count]

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }

    def predict_candidates(
        self,
        image_tensor: Tensor,
        category_names: list[str],
        box_threshold: float = 0.1,  # Lower for Task 3
        text_threshold: float = 0.1,  # Lower for Task 3
    ) -> dict[str, Tensor | list[str]]:
        """
        High-recall candidate generation for Task 3 Fusion.
        Returns as many boxes as possible for later verification.
        """
        # Reuse existing prompt logic
        cleaned_category_names = [name.strip() for name in category_names if name.strip()]
        unique_category_names = list(dict.fromkeys(cleaned_category_names))
        text_prompt = ". ".join(unique_category_names) + "."

        image_tensor_cpu = image_tensor.detach().cpu().clamp(0, 1)
        image_pil = Image.fromarray((image_tensor_cpu.permute(1, 2, 0).numpy() * 255).astype("uint8"))

        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Using much lower thresholds for the post-processor
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )[0]

        boxes = result["boxes"]  # xyxy format
        scores = result["scores"]
        labels = [str(label) for label in result.get("text_labels", result.get("labels", []))]

        # Reconcile lengths to avoid downstream mismatches
        count = min(len(boxes), len(scores), len(labels))

        return {
            "boxes": boxes[:count],
            "scores": scores[:count],
            "labels": labels[:count],
        }


class YOLOWorldHandler:
    """Wrapper for YOLO-World zero-shot object detection."""

    def __init__(
        self,
        model_id: str = "yolov8s-world.pt",
        imgsz: int = 640,
        conf: float = 0.05,
        device: DeviceSpec = "cpu",
    ) -> None:
        """Initialize YOLO-World.

        Args:
            model_id: Model checkpoint or identifier.
            imgsz: Inference image size.
            conf: Confidence threshold.
            device: PyTorch device. Defaults to CPU for YOLO-World text encoder compatibility.
        """
        self.model = YOLOWorld(model_id)
        self.imgsz = imgsz
        self.conf = conf
        self.device = resolve_torch_device(device)

        # CPU is the default/recommended workaround for the CLIP text-encoder
        # device mismatch that can occur in set_classes(...), but the device
        # remains configurable.
        self.model.to(self.device)

    def predict(self, image: Tensor, query_categories: list[str]) -> dict[str, Tensor | list[str]]:
        """Run YOLO-World on one image and return normalized predictions."""
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
            device=str(self.device),
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
        labels = [query_categories[index] for index in class_indices]

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
        }


@dataclass(frozen=True)
class QwenGenerationResult:
    """Container for raw and parser-safe Qwen generation text."""

    output_text: str
    raw_output_text: str
    thinking_text: str
    thinking_mode: bool
    assistant_prefill: str = ""
    thinking_max_new_tokens: int | None = None
    fallback_raw_output_text: str = ""
    fallback_parser_input_text: str = ""

    def debug_payload(self) -> dict[str, str | bool | int | None]:
        """Return serializable generation debug fields."""
        return {
            "raw_output_text": self.raw_output_text,
            "parser_input_text": self.output_text,
            "thinking_text": self.thinking_text,
            "thinking_mode": self.thinking_mode,
            "assistant_prefill": self.assistant_prefill,
            "thinking_max_new_tokens": self.thinking_max_new_tokens,
            "fallback_raw_output_text": self.fallback_raw_output_text,
            "fallback_parser_input_text": self.fallback_parser_input_text,
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
        "Return only concise, transferable object descriptions."
    )
    _TASK2_OBJECT_DETECTION_BOUNDED_BOXES = (
        "You are a helpful assistant to detect objects in images in a few-shot setting. "
        "You will be provided with support example images where the relevant objects "
        "are highlighted with red bounding boxes. "
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
        "You will be provided with support example images where the relevant objects "
        "are highlighted with red mark on them. "
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

    def __init__(self, model_id: str = "Qwen/Qwen3.5-2B", device: DeviceSpec = None) -> None:
        """Initialize model and processor.

        Args:
            model_id: Hugging Face model identifier.
            device: PyTorch device. Use ``None`` or ``"auto"`` for CUDA, MPS, then CPU.
        """
        self.device = resolve_torch_device(device)
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=hf_token,
            )
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                token=hf_token,
            ).to(self.device)

            # Token IDs for Task 3.B - validate single-token encoding
            def _get_single_token_id(*candidates: str) -> int:
                for candidate in candidates:
                    token_ids = self.processor.tokenizer.encode(candidate, add_special_tokens=False)
                    if len(token_ids) == 1:
                        return token_ids[0]
                raise ValueError(
                    f"Tokenizer for model_id='{model_id}' does not encode any of {candidates!r} as a single token."
                )

            self.yes_token_id = _get_single_token_id(" Yes", "Yes")
            self.no_token_id = _get_single_token_id(" No", "No")
            # Token IDs for Task 3.C
            self.a_token_id = _get_single_token_id(" A", "A")
            self.b_token_id = _get_single_token_id(" B", "B")

        except OSError as error:
            alternatives = [
                "Qwen/Qwen2-VL-2B-Instruct",
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "Qwen/Qwen2.5-VL-7B-Instruct",
                "Qwen/Qwen3.5-2B",
                "Qwen/Qwen3.5-4B",
                "Qwen/Qwen3.5-9B",
            ]
            raise OSError(
                f"Failed to load model_id='{model_id}'. This usually means the repo id is invalid or private/gated. "
                f"Try one of: {alternatives}. If you need a private model, authenticate with `hf auth login` "
                "or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
            ) from error

        self.model.eval()

    def _build_prompt(self, category_name: str, max_detections: int) -> str:
        """Build a per-request user prompt for one category."""
        return (
            f"Detect objects of class '{category_name}' in this image. "
            f"Set each detection 'label' field to '{category_name}'. "
            f"Return at most {max_detections} detection(s). "
            "Prefer high precision over high recall."
        )

    @staticmethod
    def _boxed_support_layout_instruction(support_image_count: int, support_instance_count: int) -> str:
        """Describe how boxed support examples are laid out for the VLM."""
        if support_image_count <= 1 and support_instance_count <= 1:
            return (
                "The image contains one support example. Use it to infer a likely generic object name, but avoid "
                "overfitting to incidental details from that single example. "
            )
        if support_image_count <= 1:
            return (
                f"The image contains one support image with {support_instance_count} red-boxed target instances. "
                "Compare the marked instances and describe their shared object traits. "
            )
        return (
            f"The image is a left-to-right collage of {support_image_count} source support images containing "
            f"{support_instance_count} red-boxed target instances total. Do not treat the collage as one scene or "
            "one object. Compare all marked instances and describe the shared traits that are useful for finding "
            "new instances. "
        )

    @staticmethod
    def _cropped_support_layout_instruction(support_image_count: int, support_instance_count: int) -> str:
        """Describe how cropped support examples are laid out for the VLM."""
        if support_instance_count <= 1:
            return (
                "The image contains one tightly cropped support example. Use it to infer a likely generic object "
                "name, but avoid overfitting to incidental details from that single example. "
            )
        source_note = (
            f" These crops come from {support_image_count} source support image(s)."
            if support_image_count != support_instance_count
            else ""
        )
        return (
            f"The image is a left-to-right collage of {support_instance_count} separate tightly cropped target "
            f"instances of the same object.{source_note} Compare all crops and describe the shared traits that are "
            "useful for finding new instances. "
        )

    def _build_description_prompt(self, support_image_count: int = 1, support_instance_count: int | None = None) -> str:
        """Build a prompt that asks for a support-conditioned visual description."""
        support_instance_count = support_instance_count if support_instance_count is not None else support_image_count
        return (
            f"{self._boxed_support_layout_instruction(support_image_count, support_instance_count)}"
            "Red boxes mark the target object in each support example. "
            "Write one compact object-level visual description that would help find matching instances in a new image. "
            "Start with a short generic object name inferred from appearance, then add the most useful reusable "
            "visual traits of the target object itself: overall shape, visible parts, material, texture, color, "
            "and distinctive structure. "
            "Do not include details that appear in only one support unless they seem essential to the object type. "
            "Do not describe the surrounding scene, mounting surface, position in the image, exact background, "
            "red box, or support-image context. Return one short sentence only, with no bullets or analysis."
        )

    def _build_crop_description_prompt(
        self,
        support_image_count: int = 1,
        support_instance_count: int | None = None,
    ) -> str:
        """Build a prompt that asks for a support-conditioned visual description."""
        support_instance_count = support_instance_count if support_instance_count is not None else support_image_count
        return (
            f"{self._cropped_support_layout_instruction(support_image_count, support_instance_count)}"
            "Each support example is cropped tightly around the target object. "
            "Write one compact object-level visual description that would help find matching instances in a new image. "
            "Start with a short generic object name inferred from appearance, then add the most useful reusable "
            "visual traits of the target object itself: overall shape, visible parts, material, texture, color, "
            "and distinctive structure. "
            "Do not include details that appear in only one support unless they seem essential to the object type. "
            "Return one short sentence only, with no bullets or analysis."
        )

    def _build_description_detection_prompt(
        self,
        description: str,
        max_detections: int,
    ) -> str:
        """Build a detection prompt from a generated support description."""
        return (
            "Detect objects in this image whose object-level appearance matches this visual description: "
            f"{description.strip()} "
            "Use the description as a transferable visual guide, not as a strict requirement for identical background, "
            "mounting surface, lighting, viewpoint, color shade, pose, or local context. "
            "Set each detection 'label' field to 'target_object'. "
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
        """Extract first JSON value (object or array) from model output."""
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
        """Normalize JSON detections into a unified detection dictionary format."""
        raw_detections: list[Any]
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
        """Extract detections from text like ``label(x1,y1),(x2,y2)``."""
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

    @staticmethod
    def _split_thinking_output(generated_text: str) -> tuple[str, str]:
        """Split Qwen thinking traces from the final parser-visible output."""
        stripped_text = generated_text.strip()
        final_match = re.search(r"<(?:answer|final)>(.*?)</(?:answer|final)>", stripped_text, flags=re.DOTALL)
        if final_match is not None:
            output_text = final_match.group(1).strip()
            thinking_text = stripped_text[: final_match.start()].strip()
            thinking_text = thinking_text.replace("<think>", "").replace("</think>", "").strip()
            return output_text, thinking_text

        if "</think>" in stripped_text:
            thinking_text, output_text = stripped_text.rsplit("</think>", maxsplit=1)
            thinking_text = thinking_text.replace("<think>", "").strip()
            return output_text.strip(), thinking_text

        think_block_pattern = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
        thinking_parts = [part.strip() for part in think_block_pattern.findall(stripped_text) if part.strip()]
        output_text = think_block_pattern.sub("", stripped_text).strip()
        if "<think>" in output_text:
            output_text, unfinished_thinking = output_text.split("<think>", maxsplit=1)
            if unfinished_thinking.strip():
                thinking_parts.append(unfinished_thinking.strip())

        return output_text.strip(), "\n\n".join(thinking_parts)

    @staticmethod
    def _clean_generated_text(generated_text: str) -> str:
        """Remove chat terminators while preserving thinking delimiters."""
        output_text = generated_text.strip()
        for special_token in ("<|im_start|>assistant", "<|im_end|>", "<|endoftext|>", "<|end|>"):
            output_text = output_text.replace(special_token, "")
        return output_text.strip()

    @staticmethod
    def _extract_assistant_prefill(text_prompt: str) -> str:
        """Extract assistant prefill tokens inserted by the chat template."""
        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker not in text_prompt:
            return ""
        prefill = text_prompt.rsplit(assistant_marker, maxsplit=1)[-1]
        return prefill if "<|im_end|>" not in prefill else ""

    @staticmethod
    def _close_unfinished_thinking(raw_output_text: str) -> str:
        """Return a continuation suffix that closes an unfinished thinking block."""
        if "<think>" not in raw_output_text or "</think>" in raw_output_text:
            return ""
        return "\n</think>\n\n"

    @staticmethod
    def _looks_like_unmarked_thinking(generated_text: str) -> bool:
        """Detect reasoning-style output that arrived without thinking delimiters."""
        stripped_text = generated_text.lstrip()
        lower_text = stripped_text.lower()
        if lower_text.startswith(("the user wants", "i need to", "let me", "let's", "we need to")):
            return True
        return bool(
            re.search(
                r"\n\s*\d+\.\s+\*\*(?:identify|locate|analyze|scan|re-evaluating|re-evaluate)",
                stripped_text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _with_thinking_instruction(system_prompt: str, thinking_mode: bool) -> str:
        """Add a plain-language thinking instruction for template fallbacks."""
        if thinking_mode:
            return (
                f"{system_prompt} Put any reasoning only inside <think>...</think>. After </think>, return only the "
                "requested final answer."
            )
        return f"{system_prompt} Do not include internal reasoning, chain-of-thought, or <think> blocks in the answer."

    def _parse_generated_output(
        self,
        generated_text: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Parse generated text into detections with parser diagnostics."""
        json_blob = self._extract_json_blob(generated_text)
        if isinstance(json_blob, list):
            json_detections = self._normalize_json_detections(json_blob)
            if json_detections or len(json_blob) == 0:
                return json_detections, {
                    "parser": "json",
                    "parsed_output": {"detections": json_detections},
                }
        elif isinstance(json_blob, dict):
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
        thinking_mode: bool = False,
        thinking_max_new_tokens: int | None = None,
    ) -> str:
        """Run one image-text generation step with chat-template fallback."""
        return self._generate_text_result(
            image_pil=image_pil,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            image_pils=image_pils,
            thinking_mode=thinking_mode,
            thinking_max_new_tokens=thinking_max_new_tokens,
        ).output_text

    def _generate_text_with_thinking_budget(
        self,
        image_pil: Image.Image | None,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        system_prompt: str | None,
        image_pils: list[Image.Image] | None,
        text_prompt: str,
        inputs: dict[str, torch.Tensor],
        images: list[Image.Image],
        assistant_prefill: str,
        generation_kwargs: dict[str, Any],
        thinking_max_new_tokens: int,
    ) -> QwenGenerationResult:
        """Generate thinking with its own budget, then continue with a final-answer budget."""
        thinking_generation_kwargs = {
            **generation_kwargs,
            "max_new_tokens": thinking_max_new_tokens,
        }
        with torch.no_grad():
            thinking_generated_ids = self.model.generate(**inputs, **thinking_generation_kwargs)

        if "input_ids" in inputs:
            prompt_length = inputs["input_ids"].shape[-1]
            thinking_generated_ids = thinking_generated_ids[:, prompt_length:]

        thinking_generated_text = self.processor.batch_decode(
            thinking_generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        partial_raw_output_text = self._clean_generated_text(f"{assistant_prefill}{thinking_generated_text}")
        partial_output_text, partial_thinking_text = self._split_thinking_output(partial_raw_output_text)
        if partial_output_text:
            return QwenGenerationResult(
                output_text=partial_output_text,
                raw_output_text=partial_raw_output_text,
                thinking_text=partial_thinking_text,
                thinking_mode=True,
                assistant_prefill=assistant_prefill,
                thinking_max_new_tokens=thinking_max_new_tokens,
            )

        forced_thinking_close = self._close_unfinished_thinking(partial_raw_output_text)
        continuation_prompt = f"{text_prompt}{thinking_generated_text}{forced_thinking_close}"
        continuation_inputs = self.processor(text=[continuation_prompt], images=images, return_tensors="pt")
        continuation_inputs = {key: value.to(self.device) for key, value in continuation_inputs.items()}
        with torch.no_grad():
            answer_generated_ids = self.model.generate(**continuation_inputs, **generation_kwargs)

        if "input_ids" in continuation_inputs:
            prompt_length = continuation_inputs["input_ids"].shape[-1]
            answer_generated_ids = answer_generated_ids[:, prompt_length:]

        answer_generated_text = self.processor.batch_decode(
            answer_generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        raw_output_text = self._clean_generated_text(
            f"{assistant_prefill}{thinking_generated_text}{forced_thinking_close}{answer_generated_text}"
        )
        output_text, thinking_text = self._split_thinking_output(raw_output_text)
        if not output_text:
            fallback_result = self._generate_text_result(
                image_pil=image_pil,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                image_pils=image_pils,
                thinking_mode=False,
            )
            return QwenGenerationResult(
                output_text=fallback_result.output_text,
                raw_output_text=raw_output_text,
                thinking_text=thinking_text or partial_thinking_text or partial_raw_output_text or raw_output_text,
                thinking_mode=True,
                assistant_prefill=assistant_prefill,
                thinking_max_new_tokens=thinking_max_new_tokens,
                fallback_raw_output_text=fallback_result.raw_output_text,
                fallback_parser_input_text=fallback_result.output_text,
            )
        return QwenGenerationResult(
            output_text=output_text,
            raw_output_text=raw_output_text,
            thinking_text=thinking_text or partial_thinking_text or partial_raw_output_text,
            thinking_mode=True,
            assistant_prefill=assistant_prefill,
            thinking_max_new_tokens=thinking_max_new_tokens,
        )

    def _generate_text_result(
        self,
        image_pil: Image.Image | None = None,
        prompt: str = "",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        image_pils: list[Image.Image] | None = None,
        thinking_mode: bool = False,
        thinking_max_new_tokens: int | None = None,
    ) -> QwenGenerationResult:
        """Run one image-text generation step and preserve raw thinking traces."""
        resolved_system_prompt = self._with_thinking_instruction(
            system_prompt=system_prompt or self._SYSTEM_PROMPT_OBJECT_DETECTION,
            thinking_mode=thinking_mode,
        )
        assistant_prefill = ""
        used_chat_template = False
        images = image_pils if image_pils is not None else ([image_pil] if image_pil is not None else [])
        if not images:
            raise ValueError("At least one image must be provided.")
        try:
            messages = [
                {
                    "role": "system",
                    "content": resolved_system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": image} for image in images],
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            try:
                text_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking_mode,
                )
            except TypeError:
                text_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            assistant_prefill = self._extract_assistant_prefill(text_prompt)
            inputs = self.processor(text=[text_prompt], images=images, return_tensors="pt")
            used_chat_template = True
        except (AttributeError, TypeError, ValueError):
            fallback_prompt = f"{resolved_system_prompt}\n\n{prompt}"
            inputs = self.processor(images=images, text=fallback_prompt, return_tensors="pt")

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature

        if thinking_mode and thinking_max_new_tokens is not None and assistant_prefill.startswith("<think>"):
            return self._generate_text_with_thinking_budget(
                image_pil=image_pil,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                image_pils=image_pils,
                text_prompt=text_prompt,
                inputs=inputs,
                images=images,
                assistant_prefill=assistant_prefill,
                generation_kwargs=generation_kwargs,
                thinking_max_new_tokens=thinking_max_new_tokens,
            )

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        if used_chat_template and "input_ids" in inputs:
            prompt_length = inputs["input_ids"].shape[-1]
            generated_ids = generated_ids[:, prompt_length:]

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        generated_text_prefix = assistant_prefill if thinking_mode else ""
        raw_output_text = self._clean_generated_text(f"{generated_text_prefix}{generated_text}")
        output_text, thinking_text = self._split_thinking_output(raw_output_text)
        if thinking_mode and not thinking_text and self._looks_like_unmarked_thinking(output_text):
            thinking_text = output_text
            output_text = ""
        if thinking_mode and not output_text:
            fallback_result = self._generate_text_result(
                image_pil=image_pil,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                image_pils=image_pils,
                thinking_mode=False,
            )
            return QwenGenerationResult(
                output_text=fallback_result.output_text,
                raw_output_text=raw_output_text,
                thinking_text=thinking_text or raw_output_text,
                thinking_mode=thinking_mode,
                assistant_prefill=assistant_prefill,
                thinking_max_new_tokens=thinking_max_new_tokens,
                fallback_raw_output_text=fallback_result.raw_output_text,
                fallback_parser_input_text=fallback_result.output_text,
            )
        return QwenGenerationResult(
            output_text=output_text,
            raw_output_text=raw_output_text,
            thinking_text=thinking_text,
            thinking_mode=thinking_mode,
            assistant_prefill=assistant_prefill,
            thinking_max_new_tokens=thinking_max_new_tokens,
        )

    @staticmethod
    def _normalize_description(description: str) -> str:
        """Normalize a generated support description for reuse as detector text."""
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
        """Extract sanitized detections and convert from 0-1000 reference coordinates to pixels."""
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
        thinking_mode: bool = False,
        thinking_max_new_tokens: int | None = None,
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
            thinking_mode: Enable Qwen thinking mode while stripping thinking traces before parsing.
            thinking_max_new_tokens: Optional token budget for the thinking phase before final-answer generation.

        Returns:
            Dictionary with keys: boxes (cxcywh), scores, labels.
            If ``return_debug_outputs=True``, includes ``debug_entries``.
        """
        image_tensor_cpu = image_tensor.detach().cpu().clamp(0, 1)
        image_pil = Image.fromarray((image_tensor_cpu.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        image_width, image_height = image_pil.size

        cleaned_names = [name.strip() for name in category_names if name.strip()]
        unique_category_names = list(dict.fromkeys(cleaned_names))

        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []
        debug_entries: list[dict[str, Any]] = []

        for category_name in unique_category_names:
            prompt = self._build_prompt(
                category_name=category_name,
                max_detections=max_detections_per_category,
            )
            generation_result = self._generate_text_result(
                image_pil=image_pil,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                system_prompt=system_prompt,
                thinking_mode=thinking_mode,
                thinking_max_new_tokens=thinking_max_new_tokens,
            )
            generated_text = generation_result.output_text
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
                        **generation_result.debug_payload(),
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
        thinking_mode: bool = False,
        thinking_max_new_tokens: int | None = None,
        cropped_support: bool = False,
        support_count: int | None = None,
        support_image_count: int = 1,
        support_instance_count: int | None = None,
    ) -> str:
        """Generate a concise visual description from a support example image."""
        description, _ = self.generate_support_description_debug(
            support_image=support_image,
            category_name=category_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            thinking_mode=thinking_mode,
            thinking_max_new_tokens=thinking_max_new_tokens,
            cropped_support=cropped_support,
            support_count=support_count,
            support_image_count=support_image_count,
            support_instance_count=support_instance_count,
        )
        return description

    def generate_support_description_debug(
        self,
        support_image: Image.Image,
        category_name: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        system_prompt: str | None = None,
        thinking_mode: bool = False,
        thinking_max_new_tokens: int | None = None,
        cropped_support: bool = False,
        support_count: int | None = None,
        support_image_count: int = 1,
        support_instance_count: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a visual support description with raw generation debug fields."""
        if support_count is not None:
            support_image_count = support_count
        support_instance_count = support_instance_count if support_instance_count is not None else support_image_count
        prompt = (
            self._build_crop_description_prompt(
                support_image_count=support_image_count,
                support_instance_count=support_instance_count,
            )
            if cropped_support
            else self._build_description_prompt(
                support_image_count=support_image_count,
                support_instance_count=support_instance_count,
            )
        )
        generation_result = self._generate_text_result(
            image_pil=support_image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt or self._SYSTEM_PROMPT_VISUAL_DESCRIPTION,
            thinking_mode=thinking_mode,
            thinking_max_new_tokens=thinking_max_new_tokens,
        )
        description = self._normalize_description(generation_result.output_text)
        return description, {
            "category_name": category_name,
            "cropped_support": cropped_support,
            "support_count": support_count or support_image_count,
            "support_image_count": support_image_count,
            "support_instance_count": support_instance_count,
            "input_prompt": prompt,
            "normalized_description": description,
            **generation_result.debug_payload(),
        }

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
        thinking_mode: bool = False,
        thinking_max_new_tokens: int | None = None,
    ) -> dict[str, Tensor | list[str] | list[dict[str, Any]]]:
        """Run prompt-based localization using a free-text support description."""
        normalized_description = self._normalize_description(description)
        image_tensor_cpu = image_tensor.detach().cpu().clamp(0, 1)
        image_pil = Image.fromarray((image_tensor_cpu.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        image_width, image_height = image_pil.size

        prompt = self._build_description_detection_prompt(
            description=normalized_description,
            max_detections=max_detections,
        )
        generation_result = self._generate_text_result(
            image_pil=image_pil,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
            thinking_mode=thinking_mode,
            thinking_max_new_tokens=thinking_max_new_tokens,
        )
        generated_text = generation_result.output_text
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
                    **generation_result.debug_payload(),
                    "parsed_output": parse_metadata,
                    "final_output": {
                        "boxes_cxcywh": boxes,
                        "scores": scores,
                        "labels": labels,
                    },
                }
            ]

        return outputs

    def verify_crops(self, crops, support_images, category_name):
        if not crops:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        # 1. Broaden Token Vocabulary for Robustness
        def get_valid_ids(words):
            valid_ids = []
            for w in words:
                tokens = self.processor.tokenizer.encode(w, add_special_tokens=False)
                if len(tokens) == 1:
                    valid_ids.append(tokens[0])
            return list(set(valid_ids))

        yes_ids = get_valid_ids(["Yes", " Yes", "yes", " yes", "YES", " YES"])
        no_ids = get_valid_ids(["No", " No", "no", " no", "NO", " NO"])
        if not yes_ids: yes_ids = [self.yes_token_id]
        if not no_ids: no_ids = [self.no_token_id]

        scores = []

        from torchvision.transforms import functional as TF
        
        for crop in crops:
            crop_cpu = crop.detach().cpu() if crop.device.type != "cpu" else crop
            crop_pil = TF.to_pil_image(crop_cpu)
            all_images = [*support_images, crop_pil]

            user_content = [
                {"type": "text", "text": f"Here are reference crops of the target object ({category_name}):\n"}
            ]
            for img in support_images:
                user_content.append({"type": "image", "image": img})
            
            user_content.append({"type": "text", "text": "\nFirst, describe the defining visual characteristics (core shape, parts, details) of the object shown in the references.\nNow look at this candidate crop:\n"})
            user_content.append({"type": "image", "image": crop_pil})
            user_content.append({"type": "text", "text": f"\nSecond, describe the object present in the candidate crop. Finally, carefully compare them to determine if the candidate crop is an instance of the '{category_name}' category shown in the references. It must structurally share the core defining characteristics, but details may vary. Do not write drafts or lists. Summarize your thoughts in a short paragraph, then state your final conclusion as exactly 'Yes' or 'No'."})

            messages = [
                {"role": "system", "content": "You are a precise visual evaluator."},
                {"role": "user", "content": user_content}
            ]
            
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], images=all_images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    repetition_penalty=1.1
                )
            
            input_len = inputs["input_ids"].shape[1]
            raw_reasoning_text = self.processor.decode(generated_ids[0, input_len:], skip_special_tokens=True).strip()
            
            output_text, thinking_text = self._split_thinking_output(raw_reasoning_text)
            # Fallback if split fails or outputs nothing
            if not output_text and not thinking_text:
                output_text = raw_reasoning_text
            
            reasoning_text = raw_reasoning_text

            # Pass 2: Extract binary score based on the reasoning context
            # We append the model's generated text directly into the chat log as an assistant message
            scoring_messages = [
                {"role": "system", "content": "You are a precise visual evaluator."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": raw_reasoning_text},
                {"role": "user", "content": f"Based on this analysis, does the candidate share the core defining characteristics of the '{category_name}'? Answer only Yes or No."}
            ]
            
            scoring_prompt_text = self.processor.apply_chat_template(scoring_messages, tokenize=False, add_generation_prompt=True)
            inputs_scoring = self.processor(text=[scoring_prompt_text], images=all_images, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs_scoring)
                logits = outputs.logits[0, -1, :]
                
                yes_score = torch.logsumexp(logits[yes_ids], dim=0)
                no_score = torch.logsumexp(logits[no_ids], dim=0)

                binary_probs = torch.softmax(torch.stack([yes_score, no_score]), dim=0)
                yes_prob = binary_probs[0].item()
                scores.append(yes_prob)

            # Save the first crop and its reasoning as a debug sample
            if len(scores) == 1:
                import os
                os.makedirs("debug_verify_crops", exist_ok=True)
                safe_name = category_name.replace(" ", "_").replace("/", "_")
                crop_pil.save(f"debug_verify_crops/debug_{safe_name}.png")
                with open(f"debug_verify_crops/debug_{safe_name}.txt", "w", encoding="utf-8") as f:
                    f.write(f"Category: {category_name}\n")
                    if thinking_text:
                        f.write(f"Thinking Traces:\n{thinking_text}\n\n")
                    f.write(f"Reasoning:\n{output_text}\n\n")
                    f.write(f"Final YES Probability: {yes_prob:.4f}\n")

        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    def nms_duel(self, crop_a, crop_b, category_name):
        """
        Takes two overlapping crops. Asks the VLM which one frames the object better.
        Returns 'A' or 'B'.
        """
        prompt = (
            f"<|image_pad|> This is Image A.\n"
            f"<|image_pad|> This is Image B.\n"
            f"Both images show a {category_name}. Which image frames the entire object better without cutting it off? "
            f"Answer only A or B."
        )

        crop_a_cpu = crop_a.detach().cpu() if crop_a.device.type != "cpu" else crop_a
        crop_b_cpu = crop_b.detach().cpu() if crop_b.device.type != "cpu" else crop_b
        img_a = TF.to_pil_image(crop_a_cpu)
        img_b = TF.to_pil_image(crop_b_cpu)

        inputs = self.processor(text=[prompt], images=[img_a, img_b], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]

            # Isolated Binary Softmax for A/B
            a_score = logits[self.a_token_id].item()
            b_score = logits[self.b_token_id].item()

            # Return the winning token
            return "A" if a_score > b_score else "B"

    def generate_crop_support_description(
        self,
        support_image: Image.Image,
        category_name: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        thinking_mode: bool = False,
        thinking_max_new_tokens: int | None = None,
    ) -> str:
        """Generate a visual description specifically from cropped support examples."""
        del category_name  # Currently unused in the prompt, but could be integrated for more specific descriptions.

        # We leverage the internal _generate_text helper which handles
        # the processor, device routing, and token slicing automatically.
        raw_description = self._generate_text(
            image_pil=support_image,
            prompt=self._build_crop_description_prompt(),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=self._SYSTEM_PROMPT_VISUAL_DESCRIPTION,
            thinking_mode=thinking_mode,
            thinking_max_new_tokens=thinking_max_new_tokens,
        )

        # Clean up conversational fillers (e.g., "The image shows...")
        # to provide a clean string for Grounding DINO.
        return self._normalize_description(raw_description)

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
        thinking_mode: bool = False,
        thinking_max_new_tokens: int | None = None,
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
        generation_result = self._generate_text_result(
            image_pil=support_query_panel,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=resolved_system_prompt,
            thinking_mode=thinking_mode,
            thinking_max_new_tokens=thinking_max_new_tokens,
        )
        generated_text = generation_result.output_text
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
                    **generation_result.debug_payload(),
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
        thinking_mode: bool = False,
        thinking_max_new_tokens: int | None = None,
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
            thinking_mode: Enable Qwen thinking mode while stripping thinking traces before parsing.
            thinking_max_new_tokens: Optional token budget for the thinking phase before final-answer generation.

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

        generation_result = self._generate_text_result(
            image_pils=[support_panel_pil, query_image_pil],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt or self._TASK2_OBJECT_DETECTION_BOUNDED_BOXES,
            thinking_mode=thinking_mode,
            thinking_max_new_tokens=thinking_max_new_tokens,
        )
        generated_text = generation_result.output_text

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
                    **generation_result.debug_payload(),
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
        thinking_mode: bool = False,
        thinking_max_new_tokens: int | None = None,
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
            thinking_mode: Enable Qwen thinking mode while stripping thinking traces before parsing.
            thinking_max_new_tokens: Optional token budget for the thinking phase before final-answer generation.

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
        generation_result = self._generate_text_result(
            image_pil=input_image_pil,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            system_prompt=system_prompt or self._TASK2_OBJECT_DETECTION_BOUNDED_BOXES,
            thinking_mode=thinking_mode,
            thinking_max_new_tokens=thinking_max_new_tokens,
        )
        generated_text = generation_result.output_text

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
                    **generation_result.debug_payload(),
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
