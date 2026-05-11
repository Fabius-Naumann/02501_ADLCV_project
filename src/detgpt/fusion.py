from pathlib import Path

import matplotlib
import torch
import torchvision.transforms.functional as TF
from loguru import logger
from torchvision.ops import box_iou

from detgpt import FIGURES_DIR
from detgpt.box_utils import cxcywh_tensor_to_xyxy
from detgpt.data import Task1DetectionDataset
from detgpt.device import DeviceSpec, resolve_torch_device
from detgpt.model import GroundingDINOHandler, QwenVLMHandler
from detgpt.support_samples import find_support_indices, side_by_side

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def debug_plot_boxes(image_tensor, boxes, category, save_path, scores=None, color="blue", secondary_boxes=None, secondary_color="red", secondary_scores=None):
    """Helper to save intermediate bounding box progression."""
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_np)

    if secondary_boxes is not None:
        for i, box in enumerate(secondary_boxes):
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor=secondary_color, facecolor="none", linestyle="--", alpha=0.5)
            ax.add_patch(rect)
            
            if secondary_scores is not None:
                ax.text(
                    x1, y2 + 15, f"{secondary_scores[i]:.2f}",
                    color="black", fontsize=8,
                    bbox={"facecolor": secondary_color, "alpha": 0.5, "edgecolor": "none"}
                )

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

        if scores is not None:
            ax.text(
                x1, y1 - 10, f"{scores[i]:.2f}",
                color="black", fontsize=10, fontweight="bold",
                bbox={"facecolor": color, "alpha": 0.8, "edgecolor": "none"}
            )

    plt.title(f"Step Debug: {category} - {len(boxes)} boxes ({color})", fontsize=16)
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.debug(f"Saved step visualization to {save_path}")


def get_support_crops_for_vlm(dataset, category_name, query_index, n_support=3, debug_dir=None):
    """
    Uses the dataset to find 'n' support images, crops the exact object,
    and returns a list of PIL Images. Optionally saves them to debug_dir.
    """
    support_pil_images = []
    support_indices = find_support_indices(dataset, category_name, query_index, n_support)

    if debug_dir:
        debug_path = Path(debug_dir) / f"support_debug_{category_name}"
        debug_path.mkdir(parents=True, exist_ok=True)

    for i, idx in enumerate(support_indices):
        image_tensor, target = dataset[idx]
        boxes_any = target.get("boxes")
        category_names = [str(n) for n in target.get("category_names", [])]

        selected_indices = [idx for idx, name in enumerate(category_names) if name == category_name]
        if not selected_indices:
            continue

        # Get the first matching box
        box = boxes_any[selected_indices[0]].unsqueeze(0)
        box_xyxy = cxcywh_tensor_to_xyxy(box).to(dtype=torch.int64)[0]

        # Crop and convert
        x_min, y_min, x_max, y_max = box_xyxy.tolist()
        cropped_tensor = image_tensor[:, y_min:y_max, x_min:x_max]
        crop_pil = TF.to_pil_image(cropped_tensor)

        support_pil_images.append(crop_pil)

        # Save for debugging
        if debug_dir:
            crop_pil.save(debug_path / f"support_{i}_idx_{idx}.png")

    if debug_dir:
        logger.info(f"Saved {len(support_pil_images)} support crops to {debug_path}")

    return support_pil_images


class FusionPipeline:
    def __init__(self, device: DeviceSpec = None):
        """Initialize the fusion pipeline with a shared PyTorch device."""
        self.device = resolve_torch_device(device)
        self.dino = GroundingDINOHandler(device=self.device)
        self.qwen = QwenVLMHandler(device=self.device)
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

    def run(self, image_tensor, category, dataset, query_index, detailed_prompt=None, debug_dir=None):
        # ---------------------------------------------------------
        # PHASE 1: DINO PROPOSALS (The Broad Query)
        # ---------------------------------------------------------
        query = detailed_prompt if detailed_prompt else f"{category}."
        candidates = self.dino.predict_candidates(image_tensor, [query], box_threshold=0.05)

        k = min(20, len(candidates["boxes"]))
        if k == 0:
            logger.info("No candidate boxes found; skipping verification.")
            return {
                "boxes": candidates["boxes"],
                "scores": candidates["scores"],
                "count": 0,
            }
        _, top_idx = torch.topk(candidates["scores"], k)
        boxes_to_verify = candidates["boxes"][top_idx]
        crops = self.extract_crops(image_tensor, boxes_to_verify)

        logger.info(f"Phase 1: DINO found {len(candidates['boxes'])} boxes. Keeping top {len(boxes_to_verify)}.")
        if debug_dir:
            debug_plot_boxes(
                image_tensor, boxes_to_verify, category, 
                Path(debug_dir) / f"debug_step1_dino_{category}.png", 
                scores=candidates["scores"][top_idx], color="blue"
            )

        # ---------------------------------------------------------
        # PHASE 2: SUBTASK (B) - VLM AS VERIFIER
        # ---------------------------------------------------------
        logger.info("--- Executing Subtask (b): Few-Shot Verification ---")
        # Inside FusionPipeline.run()
        support_images = get_support_crops_for_vlm(dataset, category, query_index, n_support=5, debug_dir=FIGURES_DIR)

        vlm_scores = self.qwen.verify_crops(crops, support_images, category_name=category)
        vlm_scores = vlm_scores.to(boxes_to_verify.device)

        # Filter out anything the VLM said "No" to
        conf_threshold = 0.9
        verified_mask = vlm_scores > conf_threshold
        verified_boxes = boxes_to_verify[verified_mask]
        verified_scores = vlm_scores[verified_mask]
        verified_crops = [crops[i] for i in range(len(crops)) if verified_mask[i]]

        logger.info(f"Phase 2: VLM Verification retained {len(verified_boxes)} out of {len(boxes_to_verify)} boxes (Threshold: {conf_threshold}).")
        if debug_dir:
            # Boxes that failed verification
            failed_mask = ~verified_mask
            failed_boxes = boxes_to_verify[failed_mask]
            failed_scores = vlm_scores[failed_mask]

            debug_plot_boxes(
                image_tensor, verified_boxes, category, 
                Path(debug_dir) / f"debug_step2_verified_{category}.png", 
                scores=verified_scores, color="orange",
                secondary_boxes=failed_boxes, secondary_color="red", secondary_scores=failed_scores
            )

        # ---------------------------------------------------------
        # PHASE 3: SUBTASK (C) - VLM-GUIDED NMS
        # ---------------------------------------------------------
        logger.info("--- Executing Subtask (c): VLM-Guided NMS ---")

        keep_indices = []
        # Start with boxes sorted by verification score
        remaining_idxs = verified_scores.argsort(descending=True).tolist()
        iou_threshold = 0.1

        while len(remaining_idxs) > 0:
            current_best = remaining_idxs.pop(0)
            keep_indices.append(current_best)

            next_remaining = []

            for other_idx in remaining_idxs:
                # Calculate Overlap
                iou = box_iou(verified_boxes[current_best].unsqueeze(0), verified_boxes[other_idx].unsqueeze(0)).item()
                logger.debug(f"Comparing Box {current_best} (IoU={iou:.2f}) with Box {other_idx} for category '{category}'.")

                if iou > iou_threshold:
                    logger.info(f"Overlap detected (IoU={iou:.2f}). Initiating VLM Duel...")
                    # They overlap! Send both crops to the VLM Referee
                    winner = self.qwen.nms_duel(verified_crops[current_best], verified_crops[other_idx], category)

                    if winner == "B":
                        logger.info("VLM chose Box B. Swapping...")
                        # Box B was framed better! Replace our current best.
                        keep_indices[-1] = other_idx
                        current_best = other_idx
                    else:
                        logger.info("VLM chose Box A. Suppressing Box B.")
                    # The loser is discarded (not added to next_remaining)
                else:
                    # No overlap, keep it for the next round
                    next_remaining.append(other_idx)

            remaining_idxs = next_remaining

        final_boxes = verified_boxes[keep_indices]
        final_scores = verified_scores[keep_indices]

        logger.info(f"Phase 3: VLM-NMS completed. Kept {len(final_boxes)} final boxes out of {len(verified_boxes)}.")
        if debug_dir:
            debug_plot_boxes(
                image_tensor, final_boxes, category, 
                Path(debug_dir) / f"debug_step3_nms_{category}.png", 
                scores=final_scores, color="#39FF14"
            )

        # Convert keep_indices from verified_boxes space back to boxes_to_verify space
        verified_to_candidate = verified_mask.nonzero(as_tuple=True)[0]
        keep_indices_in_candidates = verified_to_candidate[torch.tensor(keep_indices, dtype=torch.long)]

        return {
            "boxes": final_boxes,
            "scores": final_scores,
            "count": len(final_boxes),
            "all_boxes": boxes_to_verify,  # Red boxes
            "keep_indices": keep_indices_in_candidates,  # Green boxes (indices into boxes_to_verify)
            "vlm_scores": vlm_scores,
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
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="red", facecolor="none", alpha=0.3)
        ax.add_patch(rect)

    # 2. Overlay the FINAL kept boxes (The ones that survived VLM + NMS)
    kept_boxes = all_boxes[final_indices]
    kept_scores = scores[final_indices]

    for box, score in zip(kept_boxes, kept_scores, strict=True):
        x1, y1, x2, y2 = box.cpu().numpy()
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor="#39FF14", facecolor="none")
        ax.add_patch(rect)

        # Label with the VLM confidence score
        ax.text(
            x1,
            y1 - 10,
            f"{category}: {score:.2f}",
            color="black",
            fontsize=12,
            fontweight="bold",
            bbox={"facecolor": "#39FF14", "alpha": 0.9, "edgecolor": "none"},
        )

    plt.title("Fusion Debug: Red (Candidates) vs Green (VLM Verified)", fontsize=16)
    ax.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.success(f"Comparison debug image saved to {save_path}")


# --- Execution Block ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the Fusion Pipeline")
    parser.add_argument("--category", type=str, default="bread", help="The object category to detect.")
    parser.add_argument("--image-index", type=int, default=None, help="The dataset index to use as query image (Optional).")
    args = parser.parse_args()

    pipeline = FusionPipeline()
    dataset = Task1DetectionDataset(split="train")
    
    test_cat = args.category

    img_idx = args.image_index
    if img_idx is None:
        # Find an image that contains the test category
        query_indices = find_support_indices(dataset, test_cat, query_index=-1, n_support=1)
        if not query_indices:
            raise ValueError(f"No images found containing category '{test_cat}' in the dataset.")
        img_idx = query_indices[0]
    
    image, _ = dataset[img_idx]

    # Save the raw query image
    query_save_path = FIGURES_DIR / f"raw_query_{img_idx}.png"
    TF.to_pil_image(image).save(query_save_path)
    logger.info(f"Saved raw query image to {query_save_path}")

    # --- NEW: SUBTASK (A) INTEGRATION ---
    # Create the support collage for description generation
    support_indices = find_support_indices(dataset, test_cat, img_idx, n_support=3)
    support_samples = [dataset[idx] for idx in support_indices]

    if not support_samples:
        logger.warning(f"No support samples found for category '{test_cat}'. Falling back to default prompt.")
        prompt = f"{test_cat}."
    else:
        support_collage_path = FIGURES_DIR / f"support_collage_{img_idx}_{test_cat}.png"
        support_collage = side_by_side(
            target_img=None, 
            n_support_img=support_samples, 
            support_category_name=test_cat,
            output_path=support_collage_path
        )
        logger.info(f"Saved support collage to {support_collage_path}")

        # 2. Generate the "Optimal Query"
        logger.info("Generating optimal text prompt from support images...")
        prompt = pipeline.qwen.generate_crop_support_description(support_collage, test_cat)
        logger.info(f"VLM Generated Description: {prompt}")

    results = pipeline.run(
        image_tensor=image,
        category=test_cat,
        dataset=dataset,  # <--- ADD THIS
        query_index=img_idx,  # <--- ADD THIS
        detailed_prompt=prompt,
        debug_dir=FIGURES_DIR,
    )

    if results:
        logger.success(f"Fusion Finished! Found {results['count']} verified {test_cat}(s).")
        save_name = f"fusion_result_{img_idx}_{test_cat}.png"
        out_path = FIGURES_DIR / save_name

        visualize_fusion_comparison(
            image,
            results["all_boxes"],  # The 20 boxes before NMS
            results["keep_indices"],  # The indices that survived
            results["vlm_scores"],  # The actual VLM scores
            test_cat,
            FIGURES_DIR / f"fusion_debug_comparison_textvision_{test_cat}.png",
        )
