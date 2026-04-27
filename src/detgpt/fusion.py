import torch
from loguru import logger
from torchvision.ops import nms
import torchvision.transforms.functional as F

from detgpt.model import GroundingDINOHandler, QwenVLMHandler
from detgpt.data import Task1DetectionDataset
from detgpt import FIGURES_DIR

import matplotlib
matplotlib.use('Agg')  # MUST be before importing pyplot for HPC
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

class FusionPipeline:
    def __init__(self, device="cuda"):
        self.device = device
        self.dino = GroundingDINOHandler()
        self.qwen = QwenVLMHandler()
        logger.info("Fusion Pipeline Initialized (DINO + Qwen)")

    def extract_crops(self, image_tensor, boxes, padding=15):
        crops = []
        _, h, w = image_tensor.shape
        for box in boxes:
            x1, y1, x2, y2 = box.long()
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
            crops.append(image_tensor[:, y1:y2, x1:x2])
        return crops
    
    def run(self, image_tensor, category, detailed_prompt=None):
            """
            The main Fusion workflow for Task 3.
            """
            # Broad Querying
            # If there is not a detailed prompt available, we fall back to the category name
            query = detailed_prompt if detailed_prompt else f"{category}."
            logger.info(f"Broad Querying with: '{query}'")
            
            candidates = self.dino.predict_candidates(
                image_tensor, [query], box_threshold=0.05
            )
            
            if len(candidates["boxes"]) == 0:
                logger.warning("No candidates found by DINO.")
                return None

            # 2. Optimization (Top-K)
            # We limit to 20 crops to prevent the VLM from hanging/OOM
            k = min(20, len(candidates["boxes"]))
            top_scores, top_idx = torch.topk(candidates["scores"], k)
            boxes_to_verify = candidates["boxes"][top_idx]

            # Extract patches
            crops = self.extract_crops(image_tensor, boxes_to_verify)

            # VLM Verification
            logger.info(f"Verifying {len(crops)} crops via Qwen...")
            vlm_scores = self.qwen.verify_crops(crops, category)

            # VLM-Guided NMS
            # Use VLM scores to decide which overlapping box is actually the best
            logger.info("Performing VLM-Guided NMS...")
            keep_idx = nms(boxes_to_verify, vlm_scores, iou_threshold=0.3)
            
            final_boxes = boxes_to_verify[keep_idx]
            final_scores = vlm_scores[keep_idx]

            return {
                "boxes": final_boxes,          # The final survivors
                "scores": final_scores,         # The final scores
                "count": len(final_boxes),
                "all_boxes": boxes_to_verify,   # Add this: The 20 candidates
                "keep_indices": keep_idx,       # Add this: The indices chosen by NMS
                "vlm_scores": vlm_scores        # Add this: All 20 scores from Leona
            }
                    
def visualize_fusion_comparison(image_tensor, all_boxes, final_indices, scores, category, save_path):
    """
    Shows the 'Sea of Red' candidates vs the final 'VLM-Green' boxes.
    """
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image_np)

    # 1. Plot ALL candidate boxes (The ones that might be discarded)
    # Using a thin red line with transparency
    for box in all_boxes:
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            linewidth=1, edgecolor='red', facecolor='none', alpha=0.3
        )
        ax.add_patch(rect)

    # 2. Overlay the FINAL kept boxes (The ones that survived VLM + NMS)
    kept_boxes = all_boxes[final_indices]
    kept_scores = scores[final_indices]
    
    for box, score in zip(kept_boxes, kept_scores):
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1, 
            linewidth=3, edgecolor='#39FF14', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Label with the VLM confidence score
        ax.text(
            x1, y1 - 10, f"{category}: {score:.2f}", 
            color='black', fontsize=12, fontweight='bold',
            bbox=dict(facecolor='#39FF14', alpha=0.9, edgecolor='none')
        )

    plt.title(f"Fusion Debug: Red (Candidates) vs Green (VLM Verified)", fontsize=16)
    ax.axis('off')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    logger.success(f"Comparison debug image saved to {save_path}")

# --- Execution Block ---
if __name__ == "__main__":
    pipeline = FusionPipeline()
    dataset = Task1DetectionDataset(split="val")
    
    # Test on a specific image
    img_idx = 12 
    test_cat = "cat"
    image, _ = dataset[img_idx]

    prompt = "A transparent glass bottle containing liquid, with a metallic screw cap."

    results = pipeline.run(image, test_cat, detailed_prompt=prompt)

    if results:
        logger.success(f"Fusion Finished! Found {results['count']} verified {test_cat}(s).")
        save_name = f"fusion_result_{img_idx}_{test_cat}.png"
        out_path = FIGURES_DIR / save_name
        
        visualize_fusion_comparison(
            image, 
            results["all_boxes"],    # The 20 boxes before NMS
            results["keep_indices"], # The indices that survived
            results["vlm_scores"],   # The actual VLM scores
            test_cat, 
            FIGURES_DIR / "fusion_debug_comparison_cat.png"
        )