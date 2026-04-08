import json
import os
import re

import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoModelForImageTextToText, AutoModelForZeroShotObjectDetection, AutoProcessor

from detgpt.box_utils import xyxy_to_cxcywh


class Model(nn.Module):
    """Just a dummy model to show how to structure your code"""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class GroundingDINOHandler:
    """Wrapper for Grounding DINO zero-shot object detection."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny"):
        """Initialize model and processor."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

    def predict(self, image_tensor: Tensor, category_names: list[str], threshold: float = 0.3):
        """
        Run inference on a single image tensor.

        Args:
            image_tensor: C x H x W tensor from Task1DetectionDataset.
            category_names: List of labels to find (e.g., ["cat", "dog"]).
            threshold: Confidence threshold for detections.
        """
        # 1. Format text prompt: Grounding DINO prefers "item1 . item2 . item3 ."
        cleaned_category_names = [name.strip() for name in category_names if name.strip()]
        unique_category_names = list(dict.fromkeys(cleaned_category_names))
        text_prompt = ". ".join(unique_category_names) + "."

        # 2. Convert tensor back to PIL for the processor
        # (Processor handles normalization and resizing internally)
        image_tensor_cpu = image_tensor.detach().cpu().clamp(0, 1)
        image_pil = Image.fromarray((image_tensor_cpu.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3. Post-process to get boxes in pixel coordinates
        return self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image_pil.size[::-1]],
        )[0]  # Contains 'boxes', 'scores', 'labels'


class QwenVLMHandler:
    """Wrapper for prompt-based object localization with Qwen VLMs."""

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct") -> None:
        """Initialize model and processor.

        Args:
            model_id: Hugging Face model identifier.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

        try:
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
        except OSError as error:
            alternatives = [
                "Qwen/Qwen2-VL-2B-Instruct",
                "Qwen/Qwen2.5-VL-3B-Instruct",
                "Qwen/Qwen2.5-VL-7B-Instruct",
            ]
            raise OSError(
                f"Failed to load model_id='{model_id}'. This usually means the repo id is invalid or private/gated. "
                f"Try one of: {alternatives}. If you need a private model, authenticate with `hf auth login` "
                "or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN."
            ) from error

        self.model.eval()

    def _build_prompt(self, category_name: str, image_width: int, image_height: int, max_detections: int) -> str:
        """Build a strict structured prompt for one category."""
        return (
            "You are an object detector. "
            f"Find objects of class '{category_name}' in this image. "
            f"Image width is {image_width} pixels and height is {image_height} pixels. "
            f'Return ONLY valid JSON with this schema: {{"detections": [{{"bbox_xyxy": [x1, y1, x2, y2], '
            '"score": 0.0}}]}}. '
            "Use integer pixel coordinates, enforce x1 < x2 and y1 < y2, and keep coordinates inside image bounds. "
            f"Return at most {max_detections} detection(s). "
            'If no object is present, return {"detections": []}.'
        )

    @staticmethod
    def _extract_json_blob(generated_text: str) -> dict[str, list[dict[str, list[float] | float]]] | None:
        """Extract first JSON object from model output."""
        match = re.search(r"\{.*\}", generated_text, re.DOTALL)
        if match is None:
            return None

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        if not isinstance(parsed, dict):
            return None
        return parsed

    def _generate_text(self, image_pil: Image.Image, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Run one image-text generation step with chat-template fallback."""
        used_chat_template = False
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_pil},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(text=[text_prompt], images=[image_pil], return_tensors="pt")
            used_chat_template = True
        except (AttributeError, TypeError, ValueError):
            inputs = self.processor(images=image_pil, text=prompt, return_tensors="pt")

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        do_sample = temperature > 0
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
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

    def _extract_category_detections(
        self,
        parsed: dict[str, list[dict[str, list[float] | float]]],
        category_name: str,
        image_width: int,
        image_height: int,
        max_detections_per_category: int,
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """Extract sanitized detections for one category from parsed model output."""
        detections = parsed.get("detections", [])
        if not isinstance(detections, list):
            return [], [], []

        boxes: list[list[float]] = []
        scores: list[float] = []
        labels: list[str] = []

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

            x1 = min(max(x1, 0.0), float(image_width - 1))
            y1 = min(max(y1, 0.0), float(image_height - 1))
            x2 = min(max(x2, 0.0), float(image_width - 1))
            y2 = min(max(y2, 0.0), float(image_height - 1))
            if x2 <= x1 or y2 <= y1:
                continue

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
        return_raw_text_outputs: bool = False,
    ) -> dict[str, Tensor | list[str] | dict[str, str]]:
        """Run prompt-based localization for a list of categories.

        Args:
            image_tensor: C x H x W tensor from Task1DetectionDataset.
            category_names: List of labels to query.
            max_detections_per_category: Maximum detections returned per category.
            max_new_tokens: Generation length limit.
            temperature: Decoding temperature.
            return_raw_text_outputs: Include raw generated text per category in output.

        Returns:
            Dictionary with keys: boxes (cxcywh), scores, labels.
            If ``return_raw_text_outputs=True``, includes ``raw_text_outputs``.
        """
        image_tensor_cpu = image_tensor.detach().cpu().clamp(0, 1)
        image_pil = Image.fromarray((image_tensor_cpu.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        image_width, image_height = image_pil.size

        cleaned_names = [name.strip() for name in category_names if name.strip()]
        unique_category_names = list(dict.fromkeys(cleaned_names))

        all_boxes: list[list[float]] = []
        all_scores: list[float] = []
        all_labels: list[str] = []
        raw_text_outputs: dict[str, str] = {}

        for category_name in unique_category_names:
            prompt = self._build_prompt(
                category_name=category_name,
                image_width=image_width,
                image_height=image_height,
                max_detections=max_detections_per_category,
            )
            generated_text = self._generate_text(
                image_pil=image_pil,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            if return_raw_text_outputs:
                raw_text_outputs[category_name] = generated_text

            parsed = self._extract_json_blob(generated_text)
            if parsed is None:
                continue

            category_boxes, category_scores, category_labels = self._extract_category_detections(
                parsed=parsed,
                category_name=category_name,
                image_width=image_width,
                image_height=image_height,
                max_detections_per_category=max_detections_per_category,
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

        outputs: dict[str, Tensor | list[str] | dict[str, str]] = {
            "boxes": boxes_tensor,
            "scores": scores_tensor,
            "labels": all_labels,
        }
        if return_raw_text_outputs:
            outputs["raw_text_outputs"] = raw_text_outputs

        return outputs


if __name__ == "__main__":
    model = Model()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
