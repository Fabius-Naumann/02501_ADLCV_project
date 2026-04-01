import torch
from torch import nn, Tensor
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

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

    def predict(self, image_tensor: Tensor, category_names: list[str], threshold: float = 0.3):
        """
        Run inference on a single image tensor.
        
        Args:
            image_tensor: C x H x W tensor from Task1DetectionDataset.
            category_names: List of labels to find (e.g., ["cat", "dog"]).
            threshold: Confidence threshold for detections.
        """
        # 1. Format text prompt: Grounding DINO prefers "item1 . item2 . item3 ."
        text_prompt = ". ".join(list(set(category_names))) + "."
        
        # 2. Convert tensor back to PIL for the processor
        # (Processor handles normalization and resizing internally)
        image_pil = Image.fromarray((image_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8"))
        inputs = self.processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3. Post-process to get boxes in pixel coordinates
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[image_pil.size[::-1]]
        )[0]

        return results # Contains 'boxes', 'scores', 'labels'

if __name__ == "__main__":
    model = Model()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
