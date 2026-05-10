import unittest

from detgpt.lvis_api import _merge_annotations, merge_manifest_entries


class MergeAnnotationsTest(unittest.TestCase):
    def test_deduplicates_duplicate_annotation_ids_in_existing(self) -> None:
        """Test that _merge_annotations deduplicates duplicate annotation_id in existing."""
        existing_annotations = [
            {"annotation_id": 10, "category_name": "old dog"},
            {"annotation_id": 10, "category_name": "dog"},
            {"annotation_id": 20, "category_name": "cat"},
        ]
        new_annotations = [
            {"annotation_id": 30, "category_name": "truck"},
        ]

        merged = _merge_annotations(existing_annotations, new_annotations)

        self.assertEqual(len(merged), 3)
        annotation_ids = [ann.get("annotation_id") for ann in merged]
        self.assertEqual(annotation_ids.count(10), 1)
        self.assertEqual(merged[0]["category_name"], "old dog")

    def test_handles_missing_annotation_id_in_existing(self) -> None:
        """Test that _merge_annotations preserves annotations without annotation_id."""
        existing_annotations = [
            {"category_name": "no_id_1"},
            {"annotation_id": 10, "category_name": "dog"},
            {"category_name": "no_id_2"},
        ]
        new_annotations = []

        merged = _merge_annotations(existing_annotations, new_annotations)

        self.assertEqual(len(merged), 3)
        no_id_count = sum(1 for ann in merged if "annotation_id" not in ann)
        self.assertEqual(no_id_count, 2)

    def test_overrides_duplicate_annotation_ids_from_new(self) -> None:
        """Test that newer annotations override older ones when sharing annotation_id."""
        existing_annotations = [
            {"annotation_id": 10, "category_name": "old dog"},
        ]
        new_annotations = [
            {"annotation_id": 10, "category_name": "new dog"},
        ]

        merged = _merge_annotations(existing_annotations, new_annotations)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["category_name"], "new dog")

    def test_appends_new_annotations_without_duplicate_ids(self) -> None:
        """Test that annotations without annotation_id are appended properly."""
        existing_annotations = [
            {"annotation_id": 10, "category_name": "dog"},
        ]
        new_annotations = [
            {"category_name": "no_id_cat"},
        ]

        merged = _merge_annotations(existing_annotations, new_annotations)

        self.assertEqual(len(merged), 2)
        self.assertFalse(any("annotation_id" in ann for ann in merged if ann.get("category_name") == "no_id_cat"))


class MergeManifestEntriesTest(unittest.TestCase):
    def test_expands_manifest_without_dropping_existing_images(self) -> None:
        existing_manifest = [
            {
                "image_id": 1,
                "file_name": "dog.jpg",
                "annotations": [{"annotation_id": 10, "category_name": "dog"}],
                "num_annotations": 1,
            }
        ]
        new_manifest = [
            {
                "image_id": 2,
                "file_name": "truck.jpg",
                "annotations": [{"annotation_id": 20, "category_name": "truck"}],
                "num_annotations": 1,
            }
        ]

        merged_manifest = merge_manifest_entries(existing_manifest, new_manifest)

        self.assertEqual([entry["image_id"] for entry in merged_manifest], [1, 2])

    def test_merges_annotations_for_existing_image(self) -> None:
        existing_manifest = [
            {
                "image_id": 1,
                "file_name": "mixed.jpg",
                "annotations": [{"annotation_id": 10, "category_name": "old dog"}],
                "num_annotations": 1,
            }
        ]
        new_manifest = [
            {
                "image_id": 1,
                "file_name": "mixed.jpg",
                "annotations": [
                    {"annotation_id": 10, "category_name": "dog"},
                    {"annotation_id": 20, "category_name": "truck"},
                ],
                "num_annotations": 2,
            }
        ]

        merged_manifest = merge_manifest_entries(existing_manifest, new_manifest)

        self.assertEqual(len(merged_manifest), 1)
        self.assertEqual(merged_manifest[0]["num_annotations"], 2)
        self.assertEqual(
            [annotation["annotation_id"] for annotation in merged_manifest[0]["annotations"]],
            [10, 20],
        )
        self.assertEqual(merged_manifest[0]["annotations"][0]["category_name"], "dog")


if __name__ == "__main__":
    unittest.main()
