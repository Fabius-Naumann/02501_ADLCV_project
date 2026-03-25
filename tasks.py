import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "detgpt"
PYTHON_VERSION = "3.12"


# Project commands
@task
def prepare_dataset(
    ctx: Context,
    dataset_types: str = "train",
    category_names: str = "baboon,dragonfly,casserole",
    max_images_per_split: int = 200,
    download_images: bool = True,
    include_only_requested_category_annotations: bool = True,
) -> None:
    """Prepare LVIS annotations, optional image subset and local manifests."""
    category_argument = f' --category-names "{category_names}"' if category_names else ""
    download_argument = "--download-images" if download_images else "--no-download-images"
    annotation_filter_argument = (
        "--include-only-requested-category-annotations"
        if include_only_requested_category_annotations
        else "--include-all-image-annotations"
    )
    ctx.run(
        (
            f"uv run python -m {PROJECT_NAME}.lvis_api "
            f"--dataset-types {dataset_types} "
            f"--max-images-per-split {max_images_per_split} "
            f"{download_argument} {annotation_filter_argument}{category_argument}"
        ),
        echo=True,
        pty=not WINDOWS,
    )


@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
