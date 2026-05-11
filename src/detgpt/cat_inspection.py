from detgpt.data import Task1DetectionDataset


def list_dataset_categories(dataset):
    """Prints all unique categories found in the dataset annotations."""
    _, target = dataset[11]
    print(f"Categories present in this image 11: {target.get('category_names')}")
    all_categories = set()

    for _, target in dataset:
        category_names = target.get("category_names", [])
        for name in category_names:
            all_categories.add(name)

    sorted_cats = sorted(all_categories)

    print(f"Total Unique Categories: {len(sorted_cats)}")
    print("-" * 30)

    for cat in sorted_cats:
        print(cat)

    return sorted_cats


# Usage:
if __name__ == "__main__":
    dataset = Task1DetectionDataset(split="train")
    categories = list_dataset_categories(dataset)
