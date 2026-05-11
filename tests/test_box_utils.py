import unittest

import torch

from detgpt.box_utils import clip_xyxy_to_image


class ClipXyxyToImageTest(unittest.TestCase):
    """Tests for clipping xyxy boxes to image bounds."""

    def test_clips_tensor_box_to_image_bounds(self) -> None:
        """Tensor boxes should clip to valid integer image bounds."""
        clipped_box = clip_xyxy_to_image(
            torch.tensor([-4.2, 5.8, 130.1, 99.7]),
            width=120,
            height=80,
        )

        self.assertEqual(clipped_box, (0, 5, 120, 80))

    def test_clips_single_row_tensor_box_to_image_bounds(self) -> None:
        """Single-row tensor boxes should clip like flat box tensors."""
        clipped_box = clip_xyxy_to_image(
            torch.tensor([[-4.2, 5.8, 130.1, 99.7]]),
            width=120,
            height=80,
        )

        self.assertEqual(clipped_box, (0, 5, 120, 80))

    def test_expands_with_padding_before_clipping(self) -> None:
        """Padding should expand the crop bounds without exceeding the image."""
        clipped_box = clip_xyxy_to_image(
            [10, 12, 30, 42],
            width=50,
            height=55,
            padding=15,
        )

        self.assertEqual(clipped_box, (0, 0, 45, 55))

    def test_rejects_negative_padding(self) -> None:
        """Negative padding should fail clearly."""
        with self.assertRaises(ValueError):
            clip_xyxy_to_image([10, 12, 30, 42], width=50, height=55, padding=-1)


if __name__ == "__main__":
    unittest.main()
