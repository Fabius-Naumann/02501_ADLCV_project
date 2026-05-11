from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from loguru import logger
from torchvision.ops import box_iou

from detgpt import FIGURES_DIR
from detgpt.box_utils import clip_xyxy_to_image, cxcywh_tensor_to_xyxy
from detgpt.data import Task1DetectionDataset
from detgpt.model import GroundingDINOHandler, QwenVLMHandler
from detgpt.support_samples import find_support_indices, side_by_side

DINO_DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"
QWEN_DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"


def debug_plot_boxes(
    image_tensor: torch.Tensor,
    boxes: torch.Tensor,
    category: str,
    save_path: Path,
    scores: torch.Tensor | None = None,
    color: str = "blue",
    secondary_boxes: torch.Tensor | None = None,
    secondary_color: str = "red",
    secondary_scores: torch.Tensor | None = None,
) -> None:
    """Save intermediate bounding-box debug visualizations."""
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_np)

    secondary_scores_cpu = secondary_scores.detach().cpu().to(torch.float32) if secondary_scores is not None else None
    scores_cpu = scores.detach().cpu().to(torch.float32) if scores is not None else None

    if secondary_boxes is not None:
        for index, box in enumerate(secondary_boxes.detach().cpu().to(torch.float32).tolist()):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=1,
                edgecolor=secondary_color,
                facecolor="none",
                linestyle="--",
                alpha=0.5,
            )
            ax.add_patch(rect)

            if secondary_scores_cpu is not None and index < len(secondary_scores_cpu):
                ax.text(
                    x1,
                    y2 + 15,
                    f"{float(secondary_scores_cpu[index]):.2f}",
                    color="black",
                    fontsize=8,
                    bbox={"facecolor": secondary_color, "alpha": 0.5, "edgecolor": "none"},
                )

    boxes_cpu = boxes.detach().cpu().to(torch.float32)
    for index, box in enumerate(boxes_cpu.tolist()):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

        if scores_cpu is not None and index < len(scores_cpu):
            ax.text(
                x1,
                max(y1 - 10, 0),
                f"{float(scores_cpu[index]):.2f}",
                color="black",
                fontsize=10,
                fontweight="bold",
                bbox={"facecolor": color, "alpha": 0.8, "edgecolor": "none"},
            )

    ax.set_title(f"Step Debug: {category} - {len(boxes_cpu)} boxes ({color})", fontsize=16)
    ax.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.debug("Saved step visualization to {}", save_path)


def get_support_crops_for_vlm(
    dataset: Task1DetectionDataset,
    category_name: str,
    query_index: int,
    n_support: int = 5,
    debug_dir: Path | None = None,
) -> list:
    """Find support examples, crop the annotated object, and return PIL crops."""
    support_pil_images = []
    support_indices = find_support_indices(
        dataset=dataset,
        category_name=category_name,
        query_index=query_index,
        n_support=n_support,
    )

    debug_path = None
    if debug_dir is not None:
        debug_path = Path(debug_dir) / f"support_debug_{category_name}"
        debug_path.mkdir(parents=True, exist_ok=True)

    for support_number, sample_index in enumerate(support_indices):
        image_tensor, target = dataset[sample_index]
        boxes_any = target.get("boxes")
        category_names = [str(name) for name in target.get("category_names", [])]

        selected_indices = [index for index, name in enumerate(category_names) if name == category_name]
        if not selected_indices:
            continue

        box = boxes_any[selected_indices[0]].unsqueeze(0)
        box_xyxy = cxcywh_tensor_to_xyxy(box).to(dtype=torch.int64)[0]

        _, height, width = image_tensor.shape
        x_min, y_min, x_max, y_max = clip_xyxy_to_image(box_xyxy, width=width, height=height)

        if x_max <= x_min or y_max <= y_min:
            continue

        cropped_tensor = image_tensor[:, y_min:y_max, x_min:x_max]
        crop_pil = TF.to_pil_image(cropped_tensor.detach().cpu().clamp(0, 1))
        support_pil_images.append(crop_pil)

        if debug_path is not None:
            crop_pil.save(debug_path / f"support_{support_number}_idx_{sample_index}.png")

    if debug_path is not None:
        logger.info("Saved {} support crops to {}", len(support_pil_images), debug_path)

    return support_pil_images


class FusionPipeline:
    """Task 3 pipeline: DINO candidates + Qwen verification + VLM-guided NMS."""

    def __init__(
        self,
        dino_model_id: str = DINO_DEFAULT_MODEL_ID,
        qwen_model_id: str = QWEN_DEFAULT_MODEL_ID,
        dino_box_threshold: float = 0.05,
        dino_text_threshold: float = 0.05,
        top_k_candidates: int = 20,
        verification_threshold: float = 0.050,
        n_support: int = 5,
        nms_iou_threshold: float = 0.15,
    ) -> None:
        self.dino = GroundingDINOHandler(model_id=dino_model_id)
        self.qwen = QwenVLMHandler(model_id=qwen_model_id)
        self.dino_box_threshold = dino_box_threshold
        self.dino_text_threshold = dino_text_threshold
        self.top_k_candidates = top_k_candidates
        self.verification_threshold = verification_threshold
        self.n_support = n_support
        self.nms_iou_threshold = nms_iou_threshold
        logger.info(
            "Fusion Pipeline Initialized: DINO={} + Qwen={}",
            dino_model_id,
            qwen_model_id,
        )

    @staticmethod
    def extract_crops(image_tensor: torch.Tensor, boxes_xyxy: torch.Tensor, padding: int = 15) -> list[torch.Tensor]:
        """Crop candidate boxes from query image."""
        crops = []
        _, height, width = image_tensor.shape

        for box in boxes_xyxy:
            x1, y1, x2, y2 = clip_xyxy_to_image(box, width=width, height=height, padding=padding)

            if x2 <= x1 or y2 <= y1:
                continue

            crops.append(image_tensor[:, y1:y2, x1:x2])

        return crops

    @staticmethod
    def _empty_result(
        *,
        boxes_to_verify: torch.Tensor | None = None,
        vlm_scores: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ) -> dict[str, Any]:
        resolved_device = device or "cpu"
        all_boxes = (
            boxes_to_verify.detach().cpu().to(torch.float32)
            if boxes_to_verify is not None
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        scores = (
            vlm_scores.detach().cpu().to(torch.float32)
            if vlm_scores is not None
            else torch.zeros((0,), dtype=torch.float32)
        )
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32, device=resolved_device).cpu(),
            "scores": torch.zeros((0,), dtype=torch.float32, device=resolved_device).cpu(),
            "count": 0,
            "all_boxes": all_boxes,
            "keep_indices": torch.zeros((0,), dtype=torch.long),
            "vlm_scores": scores,
        }

    def run(  # noqa: C901
        self,
        image_tensor: torch.Tensor,
        category: str,
        dataset: Task1DetectionDataset,
        query_index: int,
        detailed_prompt: str | None = None,
        debug_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Run Task 3 fusion for one query image and one category."""
        logger.info("--- Executing Subtask (a): DINO Candidate Generation ---")

        query = detailed_prompt if detailed_prompt else category
        candidates = self.dino.predict_candidates(
            image_tensor=image_tensor,
            category_names=[query],
            box_threshold=self.dino_box_threshold,
            text_threshold=self.dino_text_threshold,
        )

        candidate_boxes = candidates["boxes"].detach().cpu().to(torch.float32)
        candidate_scores = candidates["scores"].detach().cpu().to(torch.float32)

        if len(candidate_boxes) == 0:
            logger.info("No candidate boxes found; skipping verification.")
            return self._empty_result(boxes_to_verify=candidate_boxes)

        k = min(self.top_k_candidates, len(candidate_boxes))
        _, top_indices = torch.topk(candidate_scores, k)
        boxes_to_verify = candidate_boxes[top_indices]

        if debug_dir is not None:
            debug_plot_boxes(
                image_tensor=image_tensor,
                boxes=boxes_to_verify,
                category=category,
                save_path=Path(debug_dir) / f"debug_step1_dino_{category}.png",
                scores=candidate_scores[top_indices],
                color="blue",
            )

        crops = self.extract_crops(image_tensor=image_tensor, boxes_xyxy=boxes_to_verify)
        if len(crops) == 0:
            logger.info("No valid crops extracted; skipping verification.")
            return self._empty_result(boxes_to_verify=boxes_to_verify)

        if len(crops) < len(boxes_to_verify):
            boxes_to_verify = boxes_to_verify[: len(crops)]

        logger.info("Generated {} candidate crops for verification.", len(crops))

        logger.info("--- Executing Subtask (b): Few-Shot VLM Verification ---")
        support_images = get_support_crops_for_vlm(
            dataset=dataset,
            category_name=category,
            query_index=query_index,
            n_support=self.n_support,
            debug_dir=debug_dir,
        )

        if len(support_images) == 0:
            logger.warning("No support crops found for category '{}'.", category)
            return self._empty_result(boxes_to_verify=boxes_to_verify)

        vlm_scores = self.qwen.verify_crops(
            crops=crops,
            support_images=support_images,
            category_name=category,
        )
        vlm_scores = vlm_scores.detach().cpu().to(torch.float32)

        verified_mask = vlm_scores > self.verification_threshold

        if verified_mask.sum().item() == 0 and len(vlm_scores) > 0:
            top_k = min(3, len(vlm_scores))
            top_indices = torch.topk(vlm_scores, top_k).indices
            verified_mask = torch.zeros_like(vlm_scores, dtype=torch.bool)
            verified_mask[top_indices] = True

        verified_boxes = boxes_to_verify[verified_mask]
        verified_scores = vlm_scores[verified_mask]
        verified_crops = [crops[index] for index in range(len(crops)) if bool(verified_mask[index])]

        logger.info("Verified {} / {} candidates.", len(verified_boxes), len(boxes_to_verify))

        if len(verified_boxes) == 0:
            return self._empty_result(boxes_to_verify=boxes_to_verify, vlm_scores=vlm_scores)

        if debug_dir is not None:
            failed_mask = ~verified_mask
            debug_plot_boxes(
                image_tensor=image_tensor,
                boxes=verified_boxes,
                category=category,
                save_path=Path(debug_dir) / f"debug_step2_verified_{category}.png",
                scores=verified_scores,
                color="orange",
                secondary_boxes=boxes_to_verify[failed_mask],
                secondary_color="red",
                secondary_scores=vlm_scores[failed_mask],
            )

        logger.info("--- Executing Subtask (c): VLM-Guided NMS ---")

        keep_indices: list[int] = []
        remaining_indices = verified_scores.argsort(descending=True).tolist()

        while remaining_indices:
            current_best = remaining_indices.pop(0)
            keep_indices.append(current_best)
            next_remaining = []

            for other_index in remaining_indices:
                iou = box_iou(
                    verified_boxes[current_best].unsqueeze(0),
                    verified_boxes[other_index].unsqueeze(0),
                ).item()

                if iou > self.nms_iou_threshold:
                    logger.info("Overlap detected (IoU={:.2f}). Initiating VLM duel.", iou)
                    winner = self.qwen.nms_duel(
                        verified_crops[current_best],
                        verified_crops[other_index],
                        category,
                    )

                    if winner == "B":
                        keep_indices[-1] = other_index
                        current_best = other_index
                else:
                    next_remaining.append(other_index)

            remaining_indices = next_remaining

        if len(keep_indices) == 0:
            return self._empty_result(boxes_to_verify=boxes_to_verify, vlm_scores=vlm_scores)

        keep_index_tensor = torch.tensor(keep_indices, dtype=torch.long)
        final_boxes = verified_boxes[keep_index_tensor]
        final_scores = verified_scores[keep_index_tensor]

        if debug_dir is not None:
            debug_plot_boxes(
                image_tensor=image_tensor,
                boxes=final_boxes,
                category=category,
                save_path=Path(debug_dir) / f"debug_step3_nms_{category}.png",
                scores=final_scores,
                color="#39FF14",
            )

        verified_to_candidate = verified_mask.nonzero(as_tuple=True)[0]
        keep_indices_in_candidates = verified_to_candidate[keep_index_tensor]

        return {
            "boxes": final_boxes.detach().cpu().to(torch.float32),
            "scores": final_scores.detach().cpu().to(torch.float32),
            "count": len(final_boxes),
            "all_boxes": boxes_to_verify.detach().cpu().to(torch.float32),
            "keep_indices": keep_indices_in_candidates.detach().cpu().to(torch.long),
            "vlm_scores": vlm_scores.detach().cpu().to(torch.float32),
        }


def visualize_fusion_comparison(
    image_tensor: torch.Tensor,
    all_boxes: torch.Tensor,
    final_indices: torch.Tensor,
    scores: torch.Tensor,
    category: str,
    save_path: Path,
) -> None:
    """Visualize DINO candidates in red and final VLM-verified boxes in green."""
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image_np)

    all_boxes_cpu = all_boxes.detach().cpu().to(torch.float32)
    final_indices_cpu = final_indices.detach().cpu().to(torch.long)
    scores_cpu = scores.detach().cpu().to(torch.float32)
    kept_index_set = set(final_indices_cpu.tolist())

    for candidate_index, box in enumerate(all_boxes_cpu.tolist()):
        x1, y1, x2, y2 = box
        is_kept = candidate_index in kept_index_set
        edge_color = "lime" if is_kept else "red"
        line_width = 3 if is_kept else 1
        alpha = 0.9 if is_kept else 0.35

        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=line_width,
            edgecolor=edge_color,
            facecolor="none",
            alpha=alpha,
        )
        ax.add_patch(rect)

        score = float(scores_cpu[candidate_index]) if candidate_index < len(scores_cpu) else 0.0
        ax.text(
            x1,
            max(y1 - 10, 0),
            f"{category}: {score:.2f}",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )

    ax.set_title("Task 3 Fusion: red = DINO candidates, green = VLM verified", fontsize=13)
    ax.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.success("Fusion comparison saved to {}", save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Fusion Pipeline")
    parser.add_argument("--category", type=str, default="bread", help="The object category to detect.")
    parser.add_argument(
        "--image-index", type=int, default=None, help="The dataset index to use as query image (Optional)."
    )
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

    support_indices = find_support_indices(dataset, test_cat, img_idx, n_support=3)
    support_samples = [dataset[index] for index in support_indices]

    if not support_samples:
        logger.warning(f"No support samples found for category '{test_cat}'. Falling back to default prompt.")
        prompt = None
    else:
        support_collage_path = FIGURES_DIR / f"support_collage_{img_idx}_{test_cat}.png"
        support_collage = side_by_side(
            target_img=None,
            n_support_img=support_samples,
            support_category_name=test_cat,
            output_path=support_collage_path,
        )
        logger.info(f"Saved support collage to {support_collage_path}")

        # 2. Generate the "Optimal Query"
        logger.info("Generating optimal text prompt from support images...")
        prompt = pipeline.qwen.generate_crop_support_description(support_collage, test_cat)
        logger.info(f"VLM Generated Description: {prompt}")

    result = pipeline.run(
        image_tensor=image,
        category=test_cat,
        dataset=dataset,
        query_index=img_idx,
        detailed_prompt=prompt,
        debug_dir=FIGURES_DIR,
    )

    if result is not None:
        logger.success("Fusion finished. Found {} verified {} detection(s).", result["count"], test_cat)
        visualize_fusion_comparison(
            image,
            result["all_boxes"],
            result["keep_indices"],
            result["vlm_scores"],
            test_cat,
            FIGURES_DIR / f"fusion_debug_comparison_{test_cat}.png",
        )
