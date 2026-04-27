import torch
import torchvision.transforms.functional as F
from detgpt.model import GroundingDINOHandler
from detgpt.data import Task1DetectionDataset
from detgpt import FIGURES_DIR
from loguru import logger

def generate_fusion_candidates(image_tensor, category_names, handler):
    """
    Generate a broad set of candidates.
    """
    # Generate high-recall candidates
    detections = handler.predict_candidates(
        image_tensor, 
        category_names, 
        box_threshold=0.05, # Very low to capture everything
        text_threshold=0.05
    )

    #logger.info(f"Generated {len(detections['boxes'])} candidates for categories: {category_names}")

    print(f"Generated {len(detections['boxes'])} candidates for categories: {category_names}")
    return detections

def extract_candidate_crops(image_tensor: torch.Tensor, boxes: torch.Tensor, padding: int = 10):
    """
    Extract image patches for VLM verification.
    
    Args:
        image_tensor: The original image (C, H, W).
        boxes: Tensor of xyxy coordinates from Grounding DINO.
        padding: Extra pixels to include around the box for context.
    """
    crops = []
    _, height, width = image_tensor.shape

    for box in boxes:
        # 1. Convert to integers and add a bit of padding for context
        x1, y1, x2, y2 = box.long()
        
        # Apply padding while staying within image boundaries
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        # 2. Slice the tensor: [Channels, Y_range, X_range]
        crop = image_tensor[:, y1:y2, x1:x2]
        
        # 3. Validation: Ensure the crop isn't empty (DINO can occasionally produce tiny boxes)
        if crop.shape[1] > 0 and crop.shape[2] > 0:
            crops.append(crop)

    return crops

def run_broad_query_pipeline(index=0, category="chair", save_debug=True):
    # 1. Load Data
    dataset = Task1DetectionDataset(split="val")
    image, target = dataset[index]
    
    # 2. Initialize Grounding DINO (Broad Mode)
    handler = GroundingDINOHandler()
    logger.info(f"Generating candidates for {category}...")
    
    # 3. Step 1: Broad Querying
    candidates = handler.predict_candidates(image, [category], box_threshold=0.05)
    boxes = candidates["boxes"]
    
    # 4. Step 2: Cropping
    logger.info(f"Snipping {len(boxes)} candidates into crops...")
    crops = extract_candidate_crops(image, boxes)
    
    # 5. Debug: Save the crops to see what Leona will receive
    if save_debug:
        debug_path = FIGURES_DIR / f"run_debug_{index}"
        debug_path.mkdir(parents=True, exist_ok=True)
        
        for i, crop in enumerate(crops):
            # Convert to PIL and save
            pil_crop = F.to_pil_image(crop)
            pil_crop.save(debug_path / f"candidate_{i:02d}.png")
            
        logger.info(f"Saved {len(crops)} crops to {debug_path}")
    
    return crops

if __name__ == "__main__":
    # Test on a single image
    #dataset = Task1DetectionDataset(split="val")
    #handler = GroundingDINOHandler()
   # image, target = dataset[0]
   # cats = list(set(target["category_names"]))
   # candidates = generate_fusion_candidates(image, cats, handler)
    
    run_broad_query_pipeline(index=5, category="bottle")
  