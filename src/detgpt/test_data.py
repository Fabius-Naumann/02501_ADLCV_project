from detgpt.data import Task1DetectionDataset, task1_collate_fn
from torch.utils.data import DataLoader
from pathlib import Path
from detgpt.visualize import save_detection_samples

# 1. Initialize the dataset for Task 1 (Zero-shot)
# We use the 'val' split since Task 1 is an evaluation task
dataset = Task1DetectionDataset(
    split="val", 
    to_float=True  # Converts images to [0, 1] range
)

print(f"Dataset initialized with {len(dataset)} samples.")

# 2. Access a single sample to verify
image, target = dataset[0]

print(f"Image shape: {image.shape}")  # Should be [3, H, W]
print(f"Categories in this image: {target['category_names']}")
print(f"Bounding boxes tensor: {target['boxes']}")

# 3. Create the DataLoader
data_loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=task1_collate_fn  # Essential for handling variable object counts
)

# Test the loader
images, targets = next(iter(data_loader))
print(f"Batch size: {len(images)}")

# Define where to save the test image
output_path = Path("outputs/figures")

# Run your visualization code for the first image
save_detection_samples(dataset, output_path, num_samples=1)
print(f"Check {output_path} for the rendered sample!")