from __future__ import annotations

from pathlib import Path

import typer

from detgpt import OUTPUTS_DIR
from detgpt.defrcn_baseline import save_eval_predictions_from_defrcn
from detgpt.evaluate_files import run_file_evaluation
from detgpt.fsod_export import export_manifest_to_coco, export_manifest_to_eval_json
from detgpt.tfa_baseline import save_eval_predictions_from_tfa

DEFAULT_CLASS_NAMES = [
    "cincture",
    "yoke_(animal_equipment)",
    "knocker_(on_a_door)",
    "poker_(fire_stirring_tool)",
    "pew_(church_bench)",
    "mail_slot",
    "cufflink",
    "oil_lamp",
    "gravy_boat",
    "quiche",
]


def _parse_class_names(class_names: str) -> list[str]:
    """Parse comma-separated class names into a unique ordered list."""
    seen: set[str] = set()
    parsed: list[str] = []

    for raw_name in class_names.split(","):
        name = raw_name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        parsed.append(name)

    return parsed


def _resolve_run_dir(
    baseline: str,
    shots: int,
    split: str,
    output_dir: str | None,
) -> Path:
    """Resolve output directory for one FSOD run."""
    run_dir = Path(output_dir) if output_dir is not None else OUTPUTS_DIR / "fsod" / f"{baseline}_{shots}shot_{split}"

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _convert_predictions(
    baseline: str,
    raw_predictions_path: Path,
    coco_gt_path: Path,
    eval_predictions_path: Path,
) -> None:
    """Convert baseline-specific raw predictions to the shared eval format."""
    normalized_baseline = baseline.strip().lower()

    if normalized_baseline == "tfa":
        save_eval_predictions_from_tfa(
            predictions_json_path=raw_predictions_path,
            coco_gt_json_path=coco_gt_path,
            output_eval_json_path=eval_predictions_path,
        )
        return

    if normalized_baseline == "defrcn":
        save_eval_predictions_from_defrcn(
            predictions_json_path=raw_predictions_path,
            coco_gt_json_path=coco_gt_path,
            output_eval_json_path=eval_predictions_path,
        )
        return

    raise typer.BadParameter(f"Unsupported baseline '{baseline}'. Use 'tfa' or 'defrcn'.")


def main(
    baseline: str = typer.Option(..., help="FSOD baseline: tfa or defrcn."),
    shots: int = typer.Option(..., help="Few-shot setting: 1 or 5."),
    split: str = typer.Option("train", help="Prepared manifest split to export/evaluate."),
    class_names: str = typer.Option(
        ",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated class names to export.",
    ),
    output_dir: str | None = typer.Option(
        None,
        help="Optional output directory. Defaults to outputs/fsod/<baseline>_<shots>shot_<split>/",
    ),
    export_only: bool = typer.Option(
        False,
        "--export-only/--run-eval",
        help="Only export COCO-style and eval-format ground truth, then stop.",
    ),
    raw_predictions_path: str | None = typer.Option(
        None,
        help="Path to raw COCO-style predictions JSON from TFA/DeFRCN.",
    ),
) -> None:
    """
    Export a prepared subset for FSOD baselines and optionally evaluate converted predictions.

    Workflow:
    1. Export current prepared manifest subset to COCO-style JSON.
    2. Export current prepared manifest subset to eval-format GT JSON.
    3. If raw_predictions_path is given, convert raw baseline predictions to the project's eval JSON format.
    4. Run offline evaluation on the converted predictions.
    """
    normalized_baseline = baseline.strip().lower()
    if normalized_baseline not in {"tfa", "defrcn"}:
        raise typer.BadParameter("baseline must be 'tfa' or 'defrcn'.")

    if shots not in {1, 5}:
        raise typer.BadParameter("shots must be 1 or 5.")

    normalized_split = split.strip().lower()
    if normalized_split not in {"train", "val"}:
        raise typer.BadParameter("split must be 'train' or 'val'.")

    selected_class_names = _parse_class_names(class_names)
    if not selected_class_names:
        raise typer.BadParameter("At least one class name must be provided.")

    run_dir = _resolve_run_dir(
        baseline=normalized_baseline,
        shots=shots,
        split=normalized_split,
        output_dir=output_dir,
    )

    coco_gt_path = run_dir / f"{normalized_split}_coco_gt.json"
    eval_ground_truth_path = run_dir / f"{normalized_split}_eval_gt.json"
    eval_predictions_path = run_dir / "eval_predictions.json"
    metrics_output_path = run_dir / "metrics.json"

    typer.echo(f"Exporting COCO-style ground truth to: {coco_gt_path}")
    export_manifest_to_coco(
        split=normalized_split,
        output_path=coco_gt_path,
        class_names=selected_class_names,
    )

    typer.echo(f"Exporting eval-format ground truth to: {eval_ground_truth_path}")
    export_manifest_to_eval_json(
        split=normalized_split,
        output_path=eval_ground_truth_path,
        class_names=selected_class_names,
    )

    if export_only:
        typer.echo("Export completed. Stopping because --export-only was set.")
        return

    if raw_predictions_path is None:
        raise typer.BadParameter(
            "raw_predictions_path is required unless --export-only is used. "
            "Provide a COCO-style predictions JSON from TFA/DeFRCN."
        )

    raw_predictions = Path(raw_predictions_path)
    if not raw_predictions.is_file():
        raise FileNotFoundError(f"Raw predictions file not found: {raw_predictions}")

    typer.echo(f"Converting raw {normalized_baseline} predictions to eval format...")
    _convert_predictions(
        baseline=normalized_baseline,
        raw_predictions_path=raw_predictions,
        coco_gt_path=coco_gt_path,
        eval_predictions_path=eval_predictions_path,
    )

    typer.echo("Running evaluation on converted predictions...")
    results = run_file_evaluation(
        predictions_path=eval_predictions_path,
        ground_truth_path=eval_ground_truth_path,
        output_path=metrics_output_path,
    )

    typer.echo(f"Saved converted predictions to: {eval_predictions_path}")
    typer.echo(f"Saved metrics to: {metrics_output_path}")
    typer.echo(
        "AP50={:.4f}, AP75={:.4f}, mean_AP_50_75={:.4f}".format(
            float(results["AP50"]["ap"]),
            float(results["AP75"]["ap"]),
            float(results["mean_AP_50_75"]),
        )
    )


if __name__ == "__main__":
    typer.run(main)
