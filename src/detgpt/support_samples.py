from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

from detgpt import FIGURES_DIR
from detgpt.box_utils import clip_xyxy_to_image, cxcywh_tensor_to_xyxy
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
    if type is None:
        return image_u8

    boxes_any = target.get("boxes")
    category_names_any = target.get("category_names", [])

    if not isinstance(boxes_any, Tensor) or boxes_any.numel() == 0:
        return image_u8

    category_names = [str(name) for name in category_names_any] if isinstance(category_names_any, list) else []

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


def _support_list(
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
) -> list[tuple[Tensor, dict[str, Any]]]:
    """Normalize one or more support samples to a list."""
    return [n_support_img] if isinstance(n_support_img, tuple) else list(n_support_img)


def _category_names(target: dict[str, Any]) -> list[str]:
    """Return string category names from a support target."""
    category_names_any = target.get("category_names", [])
    return [str(name) for name in category_names_any] if isinstance(category_names_any, list) else []


def _selected_category_indices(target: dict[str, Any], support_category_name: str | None = None) -> list[int]:
    """Return target indices matching the selected support category."""
    boxes_any = target.get("boxes")
    if not isinstance(boxes_any, Tensor) or boxes_any.numel() == 0:
        return []

    category_names = _category_names(target)
    if support_category_name is None:
        support_category_name = next((name for name in category_names if name.strip()), None)

    if support_category_name is None:
        return list(range(len(boxes_any)))

    return [index for index, name in enumerate(category_names) if name == support_category_name]


def count_support_instances(
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
) -> int:
    """Count target instances represented by the support samples."""
    return sum(
        len(_selected_category_indices(target, support_category_name)) for _, target in _support_list(n_support_img)
    )


def _expand_crop_bounds(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    width: int,
    height: int,
    padding_ratio: float,
    min_padding: int,
) -> tuple[int, int, int, int]:
    """Expand clipped ``xyxy`` bounds by a relative and minimum pixel padding."""
    box_width = x_max - x_min
    box_height = y_max - y_min
    x_padding = max(min_padding, round(box_width * padding_ratio))
    y_padding = max(min_padding, round(box_height * padding_ratio))
    crop_x_min = max(0, x_min - x_padding)
    crop_y_min = max(0, y_min - y_padding)
    crop_x_max = min(width, x_max + x_padding)
    crop_y_max = min(height, y_max + y_padding)
    return crop_x_min, crop_y_min, crop_x_max, crop_y_max


def _support_instance_crops(
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
    padding_ratio: float = 0.0,
    min_padding: int = 0,
    target_box_fills_crop: bool = False,
) -> list[tuple[Tensor, dict[str, Any]]]:
    """Extract support instance crops and rewrite target boxes into crop-local coordinates."""
    if padding_ratio < 0:
        raise ValueError("padding_ratio must be non-negative.")
    if min_padding < 0:
        raise ValueError("min_padding must be non-negative.")

    cropped_supports = []
    for image, target in _support_list(n_support_img):
        boxes_any = target.get("boxes")
        if not isinstance(boxes_any, Tensor) or boxes_any.numel() == 0:
            continue

        selected_indices = _selected_category_indices(target, support_category_name)
        if not selected_indices:
            continue

        category_names = _category_names(target)
        _, height, width = image.shape
        selected_boxes = boxes_any[selected_indices]
        boxes_xyxy = cxcywh_tensor_to_xyxy(selected_boxes).to(dtype=torch.int64)
        for selected_index, box_xyxy in zip(selected_indices, boxes_xyxy, strict=True):
            x_min, y_min, x_max, y_max = clip_xyxy_to_image(box_xyxy, width=width, height=height)

            if x_max <= x_min or y_max <= y_min:
                continue

            crop_x_min, crop_y_min, crop_x_max, crop_y_max = _expand_crop_bounds(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                width=width,
                height=height,
                padding_ratio=padding_ratio,
                min_padding=min_padding,
            )
            cropped_image = image[:, crop_y_min:crop_y_max, crop_x_min:crop_x_max]
            _, crop_height, crop_width = cropped_image.shape

            if target_box_fills_crop:
                crop_box = [crop_width / 2, crop_height / 2, crop_width, crop_height]
            else:
                target_width = x_max - x_min
                target_height = y_max - y_min
                crop_box = [
                    (x_min + x_max) / 2 - crop_x_min,
                    (y_min + y_max) / 2 - crop_y_min,
                    target_width,
                    target_height,
                ]

            cropped_target = {
                "boxes": torch.tensor([crop_box], dtype=boxes_any.dtype, device=boxes_any.device),
                "category_names": [
                    category_names[selected_index]
                    if selected_index < len(category_names) and category_names[selected_index].strip()
                    else support_category_name or ""
                ],
            }
            cropped_supports.append((cropped_image, cropped_target))

    return cropped_supports


def _resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    """Resize an image to a shared height while preserving aspect ratio."""
    if image.height == target_height:
        return image
    new_width = max(1, round(image.width * (target_height / image.height)))
    return image.resize((new_width, target_height), resample=Image.Resampling.BILINEAR)


def side_by_side(
    target_img: Tensor | None,
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
    output_path: Path | None = None,
    spacing: int = 8,
    type: str | None = "box",
) -> Image.Image:
    """Compose support example(s) and an optional query image into one image.

    Args:
        target_img: Optional query image tensor (``C x H x W``) without annotations.
        n_support_img: One or more support tuples of ``(image, target)``.
            Support ``target`` must contain ``boxes`` in ``cxcywh`` and may include
            ``category_names`` for label text.
        support_category_name: Class name to annotate on support panels.
        output_path: Optional path to save the combined image.
        spacing: Horizontal spacing in pixels between panels.

    Returns:
        Combined ``PIL.Image`` where support panels are left and the query is right
        when ``target_img`` is provided.
    """
    supports = _support_list(n_support_img)

    panels = [
        to_pil_image(_render_support_image(image, target, category_name=support_category_name, type=type))
        for image, target in supports
    ]
    if target_img is not None:
        panels.append(to_pil_image(_to_uint8_image(target_img)))

    if not panels:
        raise ValueError("At least one support image or a query image is required to compose panels.")

    max_height = max(panel.height for panel in panels)
    resized_panels = [_resize_to_height(panel, max_height) for panel in panels]

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
    target_img: Tensor | None,
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
    output_path: Path | None = None,
    spacing: int = 8,
    type: str | None = None,
) -> Image.Image:
    """Compose support example(s) and an optional query image, cropping support to target class.

    Args:
        target_img: Optional query image tensor (``C x H x W``) without annotations.
        n_support_img: One or more support tuples of ``(image, target)``.
            Support ``target`` must contain ``boxes`` in ``cxcywh`` and may include
            ``category_names`` for label text.
        support_category_name: Class name to annotate on support panels.
        output_path: Optional path to save the combined image.
        spacing: Horizontal spacing in pixels between panels.
        type: Optional annotation type. Defaults to no annotation because the
            crop already identifies the support instance.
    Returns:
        Combined ``PIL.Image`` with cropped support panels and an optional query image.
    """
    cropped_supports = _support_instance_crops(
        n_support_img=n_support_img,
        support_category_name=support_category_name,
        target_box_fills_crop=True,
    )

    return side_by_side(
        target_img=target_img,
        n_support_img=cropped_supports,
        support_category_name=support_category_name,
        output_path=output_path,
        spacing=spacing,
        type=type,
    )


def contextual_cropped_side_by_side(
    target_img: Tensor | None,
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
    output_path: Path | None = None,
    spacing: int = 8,
    type: str | None = "box",
    padding_ratio: float = 0.75,
    min_padding: int = 32,
) -> Image.Image:
    """Compose support example(s) using crops with local context around each target.

    Args:
        target_img: Optional query image tensor (``C x H x W``) without annotations.
        n_support_img: One or more support tuples of ``(image, target)``.
            Support ``target`` must contain ``boxes`` in ``cxcywh`` and may include
            ``category_names`` for label text.
        support_category_name: Class name to annotate on support panels.
        output_path: Optional path to save the combined image.
        spacing: Horizontal spacing in pixels between panels.
        type: Optional annotation type for the target within the contextual crop.
        padding_ratio: Fraction of the target box size to include as padding on each side.
        min_padding: Minimum number of pixels to include as padding on each side.

    Returns:
        Combined ``PIL.Image`` with contextual support crops and an optional query image.
    """
    contextual_supports = _support_instance_crops(
        n_support_img=n_support_img,
        support_category_name=support_category_name,
        padding_ratio=padding_ratio,
        min_padding=min_padding,
        target_box_fills_crop=False,
    )

    return side_by_side(
        target_img=target_img,
        n_support_img=contextual_supports,
        support_category_name=support_category_name,
        output_path=output_path,
        spacing=spacing,
        type=type,
    )


def marked_side_by_side(
    target_img: Tensor | None,
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
    output_path: Path | None = None,
    spacing: int = 8,
    type: str | None = "mark",
) -> Image.Image:
    """Compose support example(s) and an optional query image, marking target class.

    Args:
        target_img: Optional query image tensor (``C x H x W``) without annotations.
        n_support_img: One or more support tuples of ``(image, target)``.
            Support ``target`` must contain ``boxes`` in ``cxcywh`` and may include
            ``category_names`` for label text.
        support_category_name: Class name to annotate on support panels.
        output_path: Optional path to save the combined image.
        spacing: Horizontal spacing in pixels between panels.
        type: Type of marking to apply (e.g., "box", "mark").
    Returns:
        Combined ``PIL.Image`` with marked support panels and an optional query image.
    """
    marked_supports = []
    for image, target in _support_list(n_support_img):
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


def supports_to_images(
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
    type: str | None = "box",
) -> list[Image.Image]:
    """Return a list of PIL.Images for each support sample (annotated per `type`)."""
    supports = _support_list(n_support_img)
    images: list[Image.Image] = []
    for image, target in supports:
        rendered = _render_support_image(image, target, category_name=support_category_name, type=type)
        images.append(to_pil_image(rendered))
    return images


def cropped_supports_to_images(
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
) -> list[Image.Image]:
    """Return cropped PIL.Images for each support sample containing the target class."""
    return [
        to_pil_image(_to_uint8_image(cropped_image))
        for cropped_image, _ in _support_instance_crops(
            n_support_img=n_support_img,
            support_category_name=support_category_name,
            target_box_fills_crop=True,
        )
    ]


def contextual_cropped_supports_to_images(
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
    padding_ratio: float = 0.75,
    min_padding: int = 32,
) -> list[Image.Image]:
    """Return contextual cropped PIL.Images for each support sample containing the target class."""
    return [
        to_pil_image(_render_support_image(cropped_image, target, category_name=support_category_name, type="box"))
        for cropped_image, target in _support_instance_crops(
            n_support_img=n_support_img,
            support_category_name=support_category_name,
            padding_ratio=padding_ratio,
            min_padding=min_padding,
            target_box_fills_crop=False,
        )
    ]


def marked_supports_to_images(
    n_support_img: list[tuple[Tensor, dict[str, Any]]] | tuple[Tensor, dict[str, Any]],
    support_category_name: str | None = None,
) -> list[Image.Image]:
    """Return a list of PIL.Images for each support sample with a center mark."""
    supports = [n_support_img] if isinstance(n_support_img, tuple) else list(n_support_img)
    images: list[Image.Image] = []
    for image, target in supports:
        rendered = _render_support_image(image, target, category_name=support_category_name, type="mark")
        images.append(to_pil_image(rendered))
    return images


def find_support_indices(
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
    support_indices = find_support_indices(
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

    combined_contextual = contextual_cropped_side_by_side(
        target_img=query_img,
        n_support_img=support_samples,
        support_category_name=chosen_category,
        output_path=FIGURES_DIR / "support_query_contextual_cropped_side_by_side.png",
    )

    print(
        "Saved contextual cropped combined image to: "
        f"{FIGURES_DIR / 'support_query_contextual_cropped_side_by_side.png'}"
    )

    combined_marked = marked_side_by_side(
        target_img=query_img,
        n_support_img=support_samples,
        support_category_name=chosen_category,
        output_path=FIGURES_DIR / "support_query_marked_side_by_side.png",
        type="mark",
    )

    print(f"Saved marked combined image to: {FIGURES_DIR / 'support_query_marked_side_by_side.png'}")
