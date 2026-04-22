from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

from detgpt import FIGURES_DIR
from detgpt.box_utils import cxcywh_tensor_to_xyxy
from detgpt.data import Task1DetectionDataset


def _to_uint8_image(image: Tensor) -> Tensor:
    """Convert a ``C x H x W`` tensor to uint8 on CPU for drawing/serialization."""
    image_cpu = image.detach().cpu()
    if image_cpu.dtype != torch.uint8:
        image_cpu = image_cpu.clamp(0, 1) * 255.0
    return image_cpu.to(dtype=torch.uint8)


def _render_support_image(
    image: Tensor, target: dict[str, Any], category_name: str | None = None, type: str | None = "box"
) -> Tensor:
    """Render support image with annotation boxes for one target class only."""
    image_u8 = _to_uint8_image(image)
    boxes_any = target.get("boxes")
    category_names_any = target.get("category_names", [])

    if not isinstance(boxes_any, Tensor) or boxes_any.numel() == 0:
        return image_u8

    if isinstance(category_names_any, list):
        category_names = [str(name) for name in category_names_any]
    else:
        category_names = []

    if category_name is None:
        category_name = next((name for name in category_names if name.strip()), None)

    if category_name is None:
        return image_u8

    selected_indices = [index for index, name in enumerate(category_names) if name == category_name]
    if not selected_indices:
        return image_u8

    selected_boxes = boxes_any[selected_indices]
    boxes_xyxy = cxcywh_tensor_to_xyxy(selected_boxes).to(dtype=torch.int64)

    if type == "box":
        return draw_bounding_boxes(
            image=image_u8,
            boxes=boxes_xyxy,
            colors="red",
            width=3,
        )
    if type == "mark":
        # Mark the center of the boxes instead of drawing full boxes.
        centers = (boxes_xyxy[:, :2] + boxes_xyxy[:, 2:]) // 2
        marked_image = image_u8.clone()
        for center in centers:
            x, y = center.tolist()
            radius = 7
            left = max(0, x - radius)
            right = min(image_u8.shape[2], x + radius)
            top = max(0, y - radius)
            bottom = min(image_u8.shape[1], y + radius)
            marked_image[:, top:bottom, left:right] = torch.tensor([255, 0, 0], dtype=torch.uint8).view(3, 1, 1)
        return marked_image
    raise ValueError(f"Unsupported type '{type}'. Use 'box' or 'mark'.")


def _resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    """Resize an image to a shared height while preserving aspect ratio."""
    if image.height == target_height:
        return image
    new_width = max(1, int(round(image.width * (target_height / image.height))))
    return image.resize((new_width, target_height), resample=Image.Resampling.BILINEAR)


def side_by_side(
    target_img: Tensor,
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
    output_path: Path | None = None,
    spacing: int = 8,
    type: str | None = "box",
) -> Image.Image:
    """Compose support example(s) and query image into one side-by-side image.

    Args:
        target_img: Query image tensor (``C x H x W``) without annotations.
        n_support_img: One or more support tuples of ``(image, target)``.
            Support ``target`` must contain ``boxes`` in ``cxcywh`` and may include
            ``category_names`` for label text.
        support_category_name: Class name to annotate on support panels.
        output_path: Optional path to save the combined image.
        spacing: Horizontal spacing in pixels between panels.

    Returns:
        Combined ``PIL.Image`` where support panels are left and query is right.
    """
    supports = [n_support_img] if isinstance(n_support_img, tuple) else list(n_support_img)

    support_panels = [
        to_pil_image(_render_support_image(image, target, category_name=support_category_name, type=type))
        for image, target in supports
    ]
    query_panel = to_pil_image(_to_uint8_image(target_img))
    all_panels = [*support_panels, query_panel]

    max_height = max(panel.height for panel in all_panels)
    resized_panels = [_resize_to_height(panel, max_height) for panel in all_panels]

    total_width = sum(panel.width for panel in resized_panels) + spacing * (len(resized_panels) - 1)
    canvas = Image.new("RGB", (total_width, max_height), color=(255, 255, 255))

    cursor_x = 0
    for panel in resized_panels:
        canvas.paste(panel, (cursor_x, 0))
        cursor_x += panel.width + spacing

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)

    return canvas


def cropped_side_by_side(
    target_img: Tensor,
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
    output_path: Path | None = None,
    spacing: int = 8,
    type: str | None = "box",
) -> Image.Image:
    """Compose support example(s) and query image into one side-by-side image, cropping to target class

    Args:
        target_img: Query image tensor (``C x H x W``) without annotations.
        n_support_img: One or more support tuples of ``(image, target)``.
            Support ``target`` must contain ``boxes`` in ``cxcywh`` and may include
            ``category_names`` for label text.
        support_category_name: Class name to annotate on support panels.
        output_path: Optional path to save the combined image.
        spacing: Horizontal spacing in pixels between panels.
    Returns:
        Combined ``PIL.Image`` where support panels are left and query is right, cropped to target class.
    """
    cropped_supports = []
    for image, target in n_support_img if isinstance(n_support_img, list) else [n_support_img]:
        boxes_any = target.get("boxes")
        category_names_any = target.get("category_names", [])
        if not isinstance(boxes_any, Tensor) or boxes_any.numel() == 0:
            continue

        if isinstance(category_names_any, list):
            category_names = [str(name) for name in category_names_any]
        else:
            category_names = []

        if support_category_name is None:
            support_category_name = next((name for name in category_names if name.strip()), None)

        if support_category_name is None:
            continue

        selected_indices = [index for index, name in enumerate(category_names) if name == support_category_name]
        if not selected_indices:
            continue

        selected_boxes = boxes_any[selected_indices]
        boxes_xyxy = cxcywh_tensor_to_xyxy(selected_boxes).to(dtype=torch.int64)

        x_min = boxes_xyxy[:, 0].min().item()
        y_min = boxes_xyxy[:, 1].min().item()
        x_max = boxes_xyxy[:, 2].max().item()
        y_max = boxes_xyxy[:, 3].max().item()

        cropped_image = image[:, y_min:y_max, x_min:x_max]
        cropped_supports.append((cropped_image, target))

    return side_by_side(
        target_img=target_img,
        n_support_img=cropped_supports,
        support_category_name=support_category_name,
        output_path=output_path,
        spacing=spacing,
        type=type,
    )


def marked_side_by_side(
    target_img: Tensor,
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
    output_path: Path | None = None,
    spacing: int = 8,
    type: str | None = "mark",
) -> Image.Image:
    """Compose support example(s) and query image into one side-by-side image, marking target class

    Args:
        target_img: Query image tensor (``C x H x W``) without annotations.
        n_support_img: One or more support tuples of ``(image, target)``.
            Support ``target`` must contain ``boxes`` in ``cxcywh`` and may include
            ``category_names`` for label text.
        support_category_name: Class name to annotate on support panels.
        output_path: Optional path to save the combined image.
        spacing: Horizontal spacing in pixels between panels.
        type: Type of marking to apply (e.g., "box", "mark").
    Returns:
        Combined ``PIL.Image`` where support panels are left and query is right, with target class marked.
    """
    marked_supports = []
    for image, target in n_support_img if isinstance(n_support_img, list) else [n_support_img]:
        marked_image = _render_support_image(image, target, category_name=support_category_name, type=type)
        marked_supports.append((marked_image, target))

    return side_by_side(
        target_img=target_img,
        n_support_img=marked_supports,
        support_category_name=support_category_name,
        output_path=output_path,
        spacing=spacing,
        type=type,
    )


def _find_support_indices(
    dataset: Task1DetectionDataset,
    category_name: str,
    query_index: int,
    n_support: int,
) -> list[int]:
    """Find up to ``n_support`` indices containing ``category_name``, excluding query index."""
    category_name_cf = category_name.casefold()
    support_indices: list[int] = []

    for sample_index in range(len(dataset.samples)):
        if sample_index == query_index:
            continue

        annotations = dataset.samples[sample_index].get("annotations", [])
        category_names = [str(annotation.get("category_name", "")) for annotation in annotations]
        if any(name.casefold() == category_name_cf for name in category_names):
            support_indices.append(sample_index)
            if len(support_indices) >= n_support:
                break

    return support_indices


if __name__ == "__main__":
    # Example usage with the existing Task 1 dataset.
    dataset = Task1DetectionDataset(split="train", to_float=True)

    query_index = 0
    n_support = 5
    query_img, query_target = dataset[query_index]

    query_categories = [str(name) for name in query_target.get("category_names", []) if str(name).strip()]
    if not query_categories:
        raise ValueError("Query image has no categories. Choose another query_index.")

    chosen_category = query_categories[0]
    support_indices = _find_support_indices(
        dataset=dataset,
        category_name=chosen_category,
        query_index=query_index,
        n_support=n_support,
    )
    if not support_indices:
        raise ValueError(f"No support samples found for category '{chosen_category}'.")

    support_samples = [dataset[sample_index] for sample_index in support_indices]

    combined = side_by_side(
        target_img=query_img,
        n_support_img=support_samples,
        support_category_name=chosen_category,
        output_path=FIGURES_DIR / "support_query_side_by_side.png",
    )

    print(f"Category: {chosen_category}")
    print(f"Query index: {query_index}")
    print(f"Support indices: {support_indices}")
    print(f"Saved combined image to: {FIGURES_DIR / 'support_query_side_by_side.png'}")

    combined_cropped = cropped_side_by_side(
        target_img=query_img,
        n_support_img=support_samples,
        support_category_name=chosen_category,
        output_path=FIGURES_DIR / "support_query_cropped_side_by_side.png",
    )

    print(f"Saved cropped combined image to: {FIGURES_DIR / 'support_query_cropped_side_by_side.png'}")

    combined_marked = marked_side_by_side(
        target_img=query_img,
        n_support_img=support_samples,
        support_category_name=chosen_category,
        output_path=FIGURES_DIR / "support_query_marked_side_by_side.png",
        type="mark",
    )

    print(f"Saved marked combined image to: {FIGURES_DIR / 'support_query_marked_side_by_side.png'}")
