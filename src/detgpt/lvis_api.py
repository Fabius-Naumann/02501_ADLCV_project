import json
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen
from zipfile import ZipFile

import typer
from loguru import logger
from lvis import LVIS

from detgpt import PROCESSED_DIR, RAW_DIR

LVIS_ANNOTATION_URLS = {
    "train": "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
    "val": "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
}
VALID_SPLITS = set(LVIS_ANNOTATION_URLS)
IMAGE_ROOT_DIR = RAW_DIR / "images"


def _normalize_split(split: str) -> str:
    """Normalize and validate one LVIS split name."""
    normalized_split = split.strip().lower()
    if normalized_split not in VALID_SPLITS:
        raise ValueError(f"Invalid dataset type: {split}. Must be one of {sorted(VALID_SPLITS)}.")
    return normalized_split


def _normalize_split_list(splits: list[str]) -> list[str]:
    """Normalize, validate, and deduplicate split names while preserving order."""
    normalized = [_normalize_split(split) for split in splits if split.strip()]
    return list(dict.fromkeys(normalized))


def default_manifest_path(dataset_type: str) -> Path:
    """Return default processed manifest path for a split.

    Args:
        dataset_type: LVIS split, e.g. ``train`` or ``val``.

    Returns:
        Default manifest path for the split.
    """
    normalized_split = _normalize_split(dataset_type)
    return PROCESSED_DIR / f"lvis_v1_{normalized_split}_manifest.json"


class LvisAPI(LVIS):
    """Extended LVIS API with setup and safe download utilities."""

    def __init__(self, dataset_type: str = "train", auto_download_annotations: bool = True) -> None:
        """Initialize LVIS API for a split.

        Args:
            dataset_type: LVIS split. One of ``train`` or ``val``.
            auto_download_annotations: If true, download missing annotations automatically.

        Raises:
            ValueError: If the split is not supported.
            FileNotFoundError: If annotation file is missing and auto download is disabled.
        """
        self.dataset_type = _normalize_split(dataset_type)

        self.annotation_path = RAW_DIR / f"lvis_v1_{self.dataset_type}.json"
        if auto_download_annotations:
            self.ensure_annotation_file(self.dataset_type)
        elif not self.annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {self.annotation_path}")

        super().__init__(str(self.annotation_path))
        self.categories = self.load_cats(self.get_cat_ids())
        self.category_name_by_id = {category["id"]: category["name"] for category in self.categories}

    @staticmethod
    def _download_with_retries(url: str, output_path: Path, max_retries: int = 3, timeout_seconds: int = 60) -> None:
        """Download a file with retries and atomic replacement.

        Args:
            url: Source URL.
            output_path: Output file path.
            max_retries: Maximum retry attempts.
            timeout_seconds: Timeout per attempt in seconds.

        Raises:
            RuntimeError: If all retries fail.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

        for attempt in range(1, max_retries + 1):
            try:
                with urlopen(url, timeout=timeout_seconds) as response, temp_path.open("wb") as file_handle:
                    shutil.copyfileobj(response, file_handle)
                temp_path.replace(output_path)
                return
            except (TimeoutError, URLError, OSError) as exc:
                if temp_path.exists():
                    temp_path.unlink()
                if attempt == max_retries:
                    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts.") from exc
                time.sleep(min(2**attempt, 8))

    @classmethod
    def ensure_annotation_file(cls, dataset_type: str) -> Path:
        """Ensure the LVIS annotation JSON exists locally.

        Args:
            dataset_type: LVIS split. One of ``train`` or ``val``.

        Returns:
            Path to local annotation JSON.
        """
        dataset_type = _normalize_split(dataset_type)

        json_path = RAW_DIR / f"lvis_v1_{dataset_type}.json"
        if json_path.exists():
            return json_path

        archive_path = RAW_DIR / f"lvis_v1_{dataset_type}.json.zip"
        cls._download_with_retries(LVIS_ANNOTATION_URLS[dataset_type], archive_path)

        with ZipFile(archive_path, "r") as zip_file:
            expected_name = json_path.name
            members = zip_file.namelist()
            member = (
                expected_name
                if expected_name in members
                else next((name for name in members if name.endswith(expected_name)), None)
            )
            if member is None:
                raise FileNotFoundError(f"Could not find {expected_name} inside {archive_path}.")
            with zip_file.open(member) as source, json_path.open("wb") as target:
                shutil.copyfileobj(source, target)

        if archive_path.exists():
            archive_path.unlink()
        return json_path

    def get_category_ids_by_names(self, names: list[str]) -> list[int]:
        """Return LVIS category IDs matching names.

        Args:
            names: Category names to match.

        Returns:
            Matching category IDs.
        """
        normalized_names = {name.casefold() for name in names}
        return [cat["id"] for cat in self.categories if cat["name"].casefold() in normalized_names]

    def get_image_ids_by_category_ids(self, category_ids: list[int]) -> list[int]:
        """Return image IDs containing at least one target category.

        Args:
            category_ids: Category IDs to filter by.

        Returns:
            Sorted unique image IDs.
        """
        image_ids: set[int] = set()
        for category_id in category_ids:
            image_ids.update(self.cat_img_map.get(category_id, []))
        return sorted(image_ids)

    def get_image_ids_by_category_names(self, names: list[str]) -> list[int]:
        """Return image IDs containing at least one category from category names.

        Args:
            names: Category names.

        Returns:
            Sorted unique image IDs.
        """
        category_ids = self.get_category_ids_by_names(names)
        return self.get_image_ids_by_category_ids(category_ids)

    @staticmethod
    def _relative_image_path_from_url(url: str) -> Path:
        """Resolve a local relative image path from a COCO URL.

        Args:
            url: Remote image URL.

        Returns:
            Relative image path under the local image root.
        """
        parsed_path = Path(urlparse(url).path)
        if len(parsed_path.parts) >= 2:
            return Path(parsed_path.parts[-2]) / parsed_path.name
        return Path(parsed_path.name)

    @staticmethod
    def _bbox_xcycwh(
        bbox_xywh: list[float],
    ) -> dict[str, float]:
        """Convert LVIS/COCO ``[x, y, width, height]`` to center-format bbox.

        Args:
            bbox_xywh: LVIS/COCO bbox in absolute pixel coordinates.

        Returns:
            Center-format bbox dictionary with keys ``x_center``, ``y_center``, ``width`` and ``height``.
        """
        x_min, y_min, box_width, box_height = bbox_xywh
        x_center = x_min + box_width / 2.0
        y_center = y_min + box_height / 2.0

        return {
            "x_center": x_center,
            "y_center": y_center,
            "width": box_width,
            "height": box_height,
        }

    def _build_annotations_for_image(
        self,
        image: dict[str, Any],
        allowed_category_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Build annotation records for a single image.

        Args:
            image: LVIS image object.
            allowed_category_ids: Optional category-id filter. If provided, only matching annotations are kept.

        Returns:
            List of annotation dictionaries with class metadata and bbox variants.
        """
        annotation_ids = self.get_ann_ids(img_ids=[image["id"]])
        annotations = self.load_anns(annotation_ids)

        records: list[dict[str, Any]] = []
        for annotation in annotations:
            category_id = annotation["category_id"]
            if allowed_category_ids is not None and category_id not in allowed_category_ids:
                continue

            bbox_xywh = [float(value) for value in annotation.get("bbox", [0.0, 0.0, 0.0, 0.0])]
            if len(bbox_xywh) != 4:
                continue

            bbox_xywh_centered = self._bbox_xcycwh(bbox_xywh)

            records.append(
                {
                    "annotation_id": annotation["id"],
                    "category_id": category_id,
                    "category_name": self.category_name_by_id.get(category_id, ""),
                    "bbox_xywh": bbox_xywh,
                    "bbox_xywh_centered": bbox_xywh_centered,
                    "area": annotation.get("area"),
                }
            )
        return records

    def download_images(
        self,
        image_ids: list[int],
        image_root: Path = IMAGE_ROOT_DIR,
        max_retries: int = 3,
        timeout_seconds: int = 60,
    ) -> dict[str, int]:
        """Download selected images safely with retries.

        Args:
            image_ids: Image IDs to download.
            image_root: Local root for image files.
            max_retries: Maximum retry attempts per image.
            timeout_seconds: Timeout per image request.

        Returns:
            Download summary with keys ``downloaded``, ``skipped`` and ``failed``.
        """
        downloaded = 0
        skipped = 0
        failed = 0
        images = self.load_imgs(image_ids)

        label = f"[{self.dataset_type}] Image downloads"
        with typer.progressbar(images, label=label) as progress_bar:
            for image in progress_bar:
                image_url = image.get("coco_url")
                if not image_url:
                    failed += 1
                    continue

                rel_path = self._relative_image_path_from_url(image_url)
                output_path = image_root / rel_path

                if output_path.exists():
                    skipped += 1
                    continue

                try:
                    self._download_with_retries(
                        image_url,
                        output_path,
                        max_retries=max_retries,
                        timeout_seconds=timeout_seconds,
                    )
                    downloaded += 1
                except RuntimeError:
                    failed += 1

        return {"downloaded": downloaded, "skipped": skipped, "failed": failed}

    def write_category_stats(self, output_file: Path) -> None:
        """Save sorted category usage stats to a local file.

        Args:
            output_file: Output text file path.
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        sorted_categories = sorted(
            self.categories,
            key=lambda cat: len(set(self.cat_img_map.get(cat["id"], []))),
        )
        with output_file.open("w", encoding="utf-8") as file_handle:
            for category in sorted_categories:
                num_images = len(set(self.cat_img_map.get(category["id"], [])))
                file_handle.write(f"{category['id']}\t{category['name']}\t{num_images}\n")

    def build_manifest(
        self,
        image_ids: list[int],
        image_root: Path = IMAGE_ROOT_DIR,
        allowed_category_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Create a local manifest for selected images.

        Args:
            image_ids: Image IDs to include.
            image_root: Local root for image files.
            allowed_category_ids: Optional category-id filter for annotation records.

        Returns:
            Manifest entries.
        """
        manifest: list[dict[str, Any]] = []
        for image in self.load_imgs(image_ids):
            image_url = image.get("coco_url", "")
            local_path = ""
            if image_url:
                local_path = str(image_root / self._relative_image_path_from_url(image_url))
            annotations = self._build_annotations_for_image(image, allowed_category_ids=allowed_category_ids)
            manifest.append(
                {
                    "image_id": image["id"],
                    "height": image.get("height"),
                    "width": image.get("width"),
                    "coco_url": image_url,
                    "file_name": image.get("file_name"),
                    "local_path": local_path,
                    "annotations": annotations,
                    "num_annotations": len(annotations),
                }
            )
        return manifest


def prepare_dataset(
    dataset_types: list[str],
    category_names: list[str] | None = None,
    max_images_per_split: int = 200,
    download_images: bool = True,
    include_only_requested_category_annotations: bool = False,
) -> None:
    """Prepare local LVIS data in one call.

    The function downloads missing LVIS annotations, selects image IDs,
    optionally downloads matching images, and writes split manifests and
    category statistics under ``data/processed``.

    Args:
        dataset_types: Requested LVIS splits.
        category_names: Optional category-name filter.
        max_images_per_split: Max images per split. Set to ``0`` for no limit.
        download_images: Whether to download selected images.
        include_only_requested_category_annotations: Restrict exported annotations to requested categories only.
    """
    normalized_splits = _normalize_split_list(dataset_types)

    missing_splits = [split for split in normalized_splits if not (RAW_DIR / f"lvis_v1_{split}.json").exists()]
    if missing_splits:
        with typer.progressbar(missing_splits, label="Annotation downloads") as progress_bar:
            for split in progress_bar:
                LvisAPI.ensure_annotation_file(split)
    else:
        logger.info("Annotation files already available for all requested splits: {}", normalized_splits)

    if not category_names:
        logger.warning(
            "No category names were provided. All categories will be selected for matching images, and this can take "
            "longer to process."
        )

    for split in normalized_splits:
        logger.info("Preparing split '{}'", split)
        api = LvisAPI(dataset_type=split, auto_download_annotations=False)
        category_stats_path = PROCESSED_DIR / f"categories_lvis_{split}.txt"
        api.write_category_stats(category_stats_path)

        requested_category_ids: set[int] | None = None
        if category_names:
            requested_category_ids = set(api.get_category_ids_by_names(category_names))
            image_ids = api.get_image_ids_by_category_ids(sorted(requested_category_ids))
        else:
            image_ids = api.get_img_ids()

        if max_images_per_split > 0:
            image_ids = image_ids[:max_images_per_split]

        if download_images:
            summary = api.download_images(image_ids)
            logger.info(
                "[{}] downloaded={} skipped={} failed={}",
                split,
                summary["downloaded"],
                summary["skipped"],
                summary["failed"],
            )

        annotation_category_filter = (
            requested_category_ids if include_only_requested_category_annotations and requested_category_ids else None
        )
        manifest = api.build_manifest(image_ids, allowed_category_ids=annotation_category_filter)
        manifest_path = PROCESSED_DIR / f"lvis_v1_{split}_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file_handle:
            json.dump(manifest, file_handle, indent=2)
        logger.info("[{}] wrote manifest with {} entries to {}", split, len(manifest), manifest_path)


def _prepare_dataset_cli(
    dataset_types: str = typer.Option("train,val", help="Comma-separated split list."),
    category_names: str = typer.Option("", help="Comma-separated category names."),
    max_images_per_split: int = typer.Option(200, help="Maximum images per split. Use 0 for no limit."),
    download_images: bool = typer.Option(True, help="Download selected images."),
    include_only_requested_category_annotations: bool = typer.Option(
        False,
        "--include-only-requested-category-annotations/--include-all-image-annotations",
        help="Restrict manifest annotations to requested categories when category names are provided.",
    ),
) -> None:
    """CLI wrapper for local LVIS setup."""
    split_list = [split.strip() for split in dataset_types.split(",") if split.strip()]
    category_list = [name.strip() for name in category_names.split(",") if name.strip()]
    prepare_dataset(
        dataset_types=split_list,
        category_names=category_list if category_list else None,
        max_images_per_split=max_images_per_split,
        download_images=download_images,
        include_only_requested_category_annotations=include_only_requested_category_annotations,
    )


if __name__ == "__main__":
    typer.run(_prepare_dataset_cli)
