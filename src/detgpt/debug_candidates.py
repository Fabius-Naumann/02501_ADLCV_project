import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detgpt.model import GroundingDINOHandler
from detgpt.data import Task1DetectionDataset
from detgpt import FIGURES_DIR
from loguru import logger

def debug_fusion_step(dataset_index: int, category_name: str):
    # Setup
    dataset = Task1DetectionDataset(split="val")
    handler = GroundingDINOHandler()
    
    image_tensor, target = dataset[dataset_index]
    
    # Run Broad Query 
    # We pass the category name in a list to see how it's formatted
    detections = handler.predict_candidates(
        image_tensor, 
        [category_name], 
        box_threshold=0.05, 
        text_threshold=0.05
    )
    
    # --- VALIDATION ---
    text_prompt = f"{category_name.strip()}." 
    logger.info(f"VALIDATION - Text Query sent to DINO: '{text_prompt}'")
    logger.info(f"VALIDATION - Number of candidates found: {len(detections['boxes'])}")

    # --- VISUALIZATION ---
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_np)

    boxes = detections["boxes"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()

    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box
        # Create a rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, 
            linewidth=1, edgecolor='r', facecolor='none', alpha=0.6
        )
        ax.add_patch(rect)
        # Optional: Label with score to see which are "weak" candidates
        ax.text(x1, y1, f"{score:.2f}", color='white', fontsize=6, bbox=dict(facecolor='red', alpha=0.5))

    ax.set_title(f"Candidates for '{category_name}' (Count: {len(boxes)})")
    ax.axis('off')
    
    out_path = FIGURES_DIR / f"debug_candidates_{dataset_index}.png"
    plt.savefig(out_path, bbox_inches='tight')
    logger.info(f"Visualization saved to: {out_path}")

if __name__ == "__main__":
    # Test with a specific image and category
    debug_fusion_step(dataset_index=0, category_name="cat")