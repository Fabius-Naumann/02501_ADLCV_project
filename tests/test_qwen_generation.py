import unittest
from typing import Any

import torch
from PIL import Image

from detgpt.model import QwenVLMHandler


class _LegacyChatTemplateProcessor:
    """Fake processor that rejects the newer enable_thinking chat-template kwarg."""

    def __init__(self) -> None:
        self.text_inputs: list[Any] = []
        self.template_calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Apply a minimal chat template, rejecting enable_thinking like older processors."""
        del messages
        self.template_calls.append(kwargs)
        if "enable_thinking" in kwargs:
            raise TypeError("apply_chat_template() got an unexpected keyword argument 'enable_thinking'")
        return "<|im_start|>assistant\n"

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Return fake tokenized inputs."""
        self.text_inputs.append(kwargs.get("text"))
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    def batch_decode(self, generated_ids: torch.Tensor, **kwargs: Any) -> list[str]:
        """Decode generated IDs into a JSON answer."""
        del generated_ids, kwargs
        return ['[{"bbox_2d": [0, 0, 100, 100], "label": "target", "score": 1.0}]']


class _ThinkingBudgetProcessor:
    """Fake processor for exercising empty thinking-budget continuation fallback."""

    def __init__(self) -> None:
        self.decode_calls = 0

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Return a template that pre-fills thinking only when requested."""
        del messages
        if kwargs.get("enable_thinking"):
            return "<|im_start|>assistant\n<think>"
        return "<|im_start|>assistant\n"

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Return fake tokenized inputs."""
        del kwargs
        return {"input_ids": torch.tensor([[1, 2]], dtype=torch.long)}

    def batch_decode(self, generated_ids: torch.Tensor, **kwargs: Any) -> list[str]:
        """Decode thinking, empty continuation, and fallback generations."""
        del generated_ids, kwargs
        self.decode_calls += 1
        if self.decode_calls == 1:
            return ["I need to inspect the image."]
        if self.decode_calls == 2:
            return [""]
        return ['[{"bbox_2d": [10, 20, 30, 40], "label": "target", "score": 0.9}]']


class _FakeGenerateModel:
    """Fake model with a deterministic generate method."""

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """Return IDs longer than the prompt so prompt slicing is exercised."""
        input_ids = kwargs["input_ids"]
        batch_size = input_ids.shape[0]
        generated_suffix = torch.tensor([[4, 5]], dtype=torch.long).repeat(batch_size, 1)
        return torch.cat([input_ids, generated_suffix], dim=1)


def _build_handler(processor: Any) -> QwenVLMHandler:
    """Build a Qwen handler without loading external model weights."""
    handler = QwenVLMHandler.__new__(QwenVLMHandler)
    handler.device = torch.device("cpu")
    handler.processor = processor
    handler.model = _FakeGenerateModel()
    return handler


class QwenGenerationTest(unittest.TestCase):
    """Tests for Qwen generation compatibility and fallback behavior."""

    def test_retries_chat_template_without_enable_thinking_kwarg(self) -> None:
        """Older processors should still use chat-template formatting."""
        processor = _LegacyChatTemplateProcessor()
        handler = _build_handler(processor)

        result = handler._generate_text_result(
            image_pil=Image.new("RGB", (8, 8)),
            prompt="Detect the target.",
            thinking_mode=False,
        )

        self.assertIn("bbox_2d", result.output_text)
        self.assertEqual(len(processor.template_calls), 2)
        self.assertIn("enable_thinking", processor.template_calls[0])
        self.assertNotIn("enable_thinking", processor.template_calls[1])
        self.assertEqual(processor.text_inputs, [["<|im_start|>assistant\n"]])

    def test_thinking_budget_falls_back_when_continuation_has_no_answer(self) -> None:
        """Thinking-budget generation should not return an empty parser input."""
        processor = _ThinkingBudgetProcessor()
        handler = _build_handler(processor)

        result = handler._generate_text_result(
            image_pil=Image.new("RGB", (8, 8)),
            prompt="Detect the target.",
            thinking_mode=True,
            thinking_max_new_tokens=4,
        )

        self.assertIn("bbox_2d", result.output_text)
        self.assertIn("bbox_2d", result.fallback_parser_input_text)
        self.assertEqual(processor.decode_calls, 3)


if __name__ == "__main__":
    unittest.main()
