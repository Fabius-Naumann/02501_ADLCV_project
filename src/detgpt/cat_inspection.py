from detgpt.data import Task1DetectionDataset


def list_dataset_categories(dataset):
    """Prints all unique categories found in the dataset annotations."""
    image, target = dataset[11]
    print(f"Categories present in this image 11: {target.get('category_names')}")
    all_categories = set()

    for sample in dataset.samples:
        annotations = sample.get("annotations", [])
        for ann in annotations:
            name = ann.get("category_name")
            if name:
                all_categories.add(name)

    sorted_cats = sorted(list(all_categories))
    print(f"Total Unique Categories: {len(sorted_cats)}")
    print("-" * 30)
    for cat in sorted_cats:
        print(f"- {cat}")

    return sorted_cats


# Usage:
if __name__ == "__main__":
    dataset = Task1DetectionDataset(split="train")
    categories = list_dataset_categories(dataset)
