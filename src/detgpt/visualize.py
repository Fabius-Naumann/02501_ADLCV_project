from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch import Tensor
from torch.utils.data import Dataset

SHOW_PLOTS = False


def _save_or_show_figure(fig: plt.Figure, output_path: Path) -> None:
    """Save or show the current figure based on the visualization mode."""
    if SHOW_PLOTS:
        plt.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
    plt.close(fig)


def save_detection_samples(
    dataset: Dataset[tuple[Tensor, dict[str, Any]]],
    output_dir: Path,
    num_samples: int = 5,
) -> None:
    """Save sample images with center-format ``xywh`` bounding boxes.

    Args:
        dataset: Dataset returning ``(image, target)`` pairs.
        output_dir: Directory where rendered figures are written.
        num_samples: Number of first samples to render.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_index in range(min(num_samples, len(dataset))):
        image, target = dataset[sample_index]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image.permute(1, 2, 0).numpy())

        for box in target["boxes"]:
            x_center, y_center, box_width, box_height = box.tolist()
            x_min = x_center - box_width / 2
            y_min = y_center - box_height / 2
            rect = Rectangle((x_min, y_min), box_width, box_height, edgecolor="red", facecolor="none")
            ax.add_patch(rect)

        ax.set_title(f"Sample {sample_index} with {len(target['boxes'])} annotations")
        ax.axis("off")
        output_path = output_dir / f"dataset_sample_{sample_index}.png"
        _save_or_show_figure(fig, output_path)
