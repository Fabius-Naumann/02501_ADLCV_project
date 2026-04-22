from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from detgpt.lvis_api import default_manifest_path


def load_manifest(split: str = "train") -> list[dict[str, Any]]:
    """Load a prepared manifest for a split."""
    manifest_path = default_manifest_path(split)
    with manifest_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Manifest JSON must contain a list of samples.")

    return data


def save_manifest(manifest: list[dict[str, Any]], output_path: str | Path) -> None:
    """Save a manifest list to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def _normalize_class_names(class_names: list[str]) -> list[str]:
    """Deduplicate class names while preserving order."""
    seen: set[str] = set()
    normalized: list[str] = []

    for name in class_names:
        clean_name = str(name).strip()
        if not clean_name or clean_name in seen:
            continue
        seen.add(clean_name)
        normalized.append(clean_name)

    return normalized


def build_class_to_instances(
    manifest: list[dict[str, Any]],
    class_names: list[str] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """
    Build class -> list of (sample_index, annotation_index) pairs.
    """
    allowed = set(_normalize_class_names(class_names)) if class_names else None
    class_to_instances: dict[str, list[tuple[int, int]]] = {}

    for sample_index, sample in enumerate(manifest):
        annotations = sample.get("annotations", [])
        for annotation_index, annotation in enumerate(annotations):
            class_name = str(annotation.get("category_name", "")).strip()
            if not class_name:
                continue
            if allowed is not None and class_name not in allowed:
                continue

            if class_name not in class_to_instances:
                class_to_instances[class_name] = []
            class_to_instances[class_name].append((sample_index, annotation_index))

    return class_to_instances


def sample_support_indices(
    manifest: list[dict[str, Any]],
    class_names: list[str],
    shots: int,
    seed: int = 0,
) -> dict[str, list[tuple[int, int]]]:
    """
    Sample k annotation instances per class for the support set.
    """
    if shots <= 0:
        raise ValueError("shots must be positive.")

    normalized_class_names = _normalize_class_names(class_names)
    class_to_instances = build_class_to_instances(manifest, normalized_class_names)
    rng = random.Random(seed)

    support: dict[str, list[tuple[int, int]]] = {}
    for class_name in normalized_class_names:
        instances = class_to_instances.get(class_name, [])
        if len(instances) < shots:
            raise ValueError(
                f"Class '{class_name}' has only {len(instances)} available instances, "
                f"but {shots} shot(s) were requested."
            )

        support[class_name] = rng.sample(instances, shots)

    return support


def build_support_manifest(
    manifest: list[dict[str, Any]],
    support: dict[str, list[tuple[int, int]]],
) -> list[dict[str, Any]]:
    """
    Build a support manifest containing only the selected support annotations.
    """
    support_by_sample: dict[int, set[int]] = {}

    for pairs in support.values():
        for sample_index, annotation_index in pairs:
            if sample_index not in support_by_sample:
                support_by_sample[sample_index] = set()
            support_by_sample[sample_index].add(annotation_index)

    support_manifest: list[dict[str, Any]] = []
    for sample_index in sorted(support_by_sample):
        sample = manifest[sample_index]
        keep_annotation_indices = support_by_sample[sample_index]

        filtered_annotations = [
            annotation
            for annotation_index, annotation in enumerate(sample.get("annotations", []))
            if annotation_index in keep_annotation_indices
        ]

        support_sample = dict(sample)
        support_sample["annotations"] = filtered_annotations
        support_sample["num_annotations"] = len(filtered_annotations)
        support_manifest.append(support_sample)

    return support_manifest


def build_query_manifest(
    manifest: list[dict[str, Any]],
    support: dict[str, list[tuple[int, int]]],
) -> list[dict[str, Any]]:
    """
    Build the query manifest by removing all support images.
    """
    support_image_indices = {sample_index for pairs in support.values() for sample_index, _ in pairs}

    return [sample for sample_index, sample in enumerate(manifest) if sample_index not in support_image_indices]


def build_fsod_split(
    split: str,
    class_names: list[str],
    shots: int,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[tuple[int, int]]]]:
    """
    Build support/query manifests for a few-shot split.

    Returns:
        support_manifest, query_manifest, support_index_map
    """
    manifest = load_manifest(split)
    support = sample_support_indices(
        manifest=manifest,
        class_names=class_names,
        shots=shots,
        seed=seed,
    )
    support_manifest = build_support_manifest(manifest, support)
    query_manifest = build_query_manifest(manifest, support)

    return support_manifest, query_manifest, support


def save_fsod_split(
    split: str,
    class_names: list[str],
    shots: int,
    output_dir: str | Path,
    seed: int = 0,
) -> None:
    """
    Build and save support/query manifests and the sampled support index map.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    support_manifest, query_manifest, support = build_fsod_split(
        split=split,
        class_names=class_names,
        shots=shots,
        seed=seed,
    )

    save_manifest(support_manifest, output_dir / f"{split}_{shots}shot_support_manifest.json")
    save_manifest(query_manifest, output_dir / f"{split}_{shots}shot_query_manifest.json")

    support_index_payload = {
        class_name: [
            {
                "sample_index": sample_index,
                "annotation_index": annotation_index,
            }
            for sample_index, annotation_index in pairs
        ]
        for class_name, pairs in support.items()
    }
    with (output_dir / f"{split}_{shots}shot_support_indices.json").open("w", encoding="utf-8") as file:
        json.dump(support_index_payload, file, indent=2)
