import unittest

from detgpt.lvis_api import merge_manifest_entries


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
