# DetGPT: In-Context Object Detection via Visual Example Prompting
This project investigates In-Context Learning (ICL) for object detection using Vision-Language Models (VLMs) like Qwen2.5-VL and InternVL. We aim to improve few-shot detection performance by providing visual examples with bounding boxes rather than just text prompts.

## Data Pipeline

Our pipeline is divided into two distinct stages:

### Phase 1: Preparation (`lvis_api.py`)
The **Producer** stage. It handles the raw interface with the LVIS dataset.
* **Filter & Sample**: Selects specific categories and limits the number of images to create a manageable subset (e.g., for few-shot testing).
* **Download**: Pulls images and annotations from remote servers using atomic writes.
* **Manifest Generation**: Creates a local `json` manifest in `data/processed/` which serves as the "source of truth" for the models.

### Phase 2: Consumption (`data.py`)
The **Consumer** stage. It transforms processed data into GPU-ready tensors.
* **Dataset Loading**: The `Task1DetectionDataset` class reads the manifest and loads local images using `torchvision`.
* **Feature Engineering**: Converts standard LVIS bounding boxes into center-format $[x_c, y_c, w, h]$ required for VLM spatial reasoning.
* **Batching**: Implements a custom `task1_collate_fn` to handle variable numbers of objects per image.

**Usage Example:**
```bash
# 1. Prepare data (LVIS API)
uv run src/detgpt/lvis_api.py --category-names "dog,truck" --max-images-per-split 50

# 2. Verify dataset (Data Loader)
uv run src/detgpt/data.py
```

## Evaluation Scripts

The repository provides two evaluation entrypoints:

1. `detgpt.evaluate` for running model inference directly on the prepared dataset.
2. `detgpt.evaluate_files` for computing metrics from saved prediction and ground-truth JSON files.

---

### 1. Dataset Inference Evaluation (`evaluate.py`)

This script runs end-to-end evaluation: dataset loading, model inference, and metric computation.

It is implemented as a CLI using Typer.

#### Standard Evaluation Setting (Recommended)

We use class-balanced sampling to ensure fair comparison across rare LVIS categories:

```bash
--balanced --samples-per-class 10 --limit 100
```

Run Grounding Dino:

```bash
uv run python -m detgpt.evaluate \
  --detector-backend grounding_dino \
  --split train \
  --balanced \
  --samples-per-class 10 \
  --limit 100 \
  --save-results \
  --save-viz
```

Run Qwen-VLM evaluation with deterministic decoding and debug trace:

```bash
uv run python -m detgpt.evaluate \
  --detector-backend qwen_vlm \
  --model-id Qwen/Qwen3.5-2B \
  --split train \
  --balanced \
  --samples-per-class 10 \
  --limit 100 \
  --qwen-max-detections-per-category 5 \
  --qwen-temperature 0.0 \
  --qwen-debug-dump \
  --save-viz
```

Run YOLO-World:
```bash
uv run python -m detgpt.evaluate \
  --detector-backend yolo_world \
  --split train \
  --balanced \
  --samples-per-class 10 \
  --limit 100 \
  --save-results \
  --save-viz
```

Output files are written to `outputs/task1_results/run_<timestamp>/`.
When `--qwen-debug-dump` is enabled, a `qwen_debug_trace.jsonl` file is saved in the same run directory.

### 2. File-Based Metrics Evaluation (`evaluate_files.py`)

Use this when predictions are already exported as JSON (to be refined - TODO: @Alexandra):.

```bash
uv run python - <<'PY'
from detgpt.evaluate_files import run_file_evaluation

results = run_file_evaluation(
		predictions_path="outputs/eval/predictions.json",
		ground_truth_path="outputs/eval/ground_truth.json",
		output_path="outputs/eval/metrics.json",
)
print(results)
PY
```

Expected JSON record format:

```json
[
	{
		"image_path": "data/raw/images/train2017/example.jpg",
		"boxes": [[165.0, 95.0, 50.0, 10.0]],
		"labels": ["car"],
		"scores": [0.91]
	}
]
```

Note: `boxes` are expected in `cxcywh` format for both predictions and ground truth.

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── outputs/                  # local run-output folder
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Dataset

### Rare classes
Validated against `data/processed/categories_lvis_train.txt`.

For zero-shot stress testing on modern VLMs, prioritize globally long-tail classes
(niche objects, old-fashioned artifacts, and taxonomy-specific labels), not only local LVIS rarity.

Zero-shot priority shortlist (all targets have 10+ occurrences in this split):

- cincture (18): negatives -> belt, necklace, earring
- yoke_(animal_equipment) (11): negatives -> headstall_(for_horses), blinder_(for_horses), necktie
- knocker_(on_a_door) (10): negatives -> doorknob, handle, bell
- poker_(fire_stirring_tool) (14): negatives -> crowbar, screwdriver, bottle_opener
- pew_(church_bench) (14): negatives -> bench, chair
- mail_slot (15): negatives -> mailbox_(at_home), postbox_(public), envelope
- cufflink (15): negatives -> bracelet, earring, ring
- oil_lamp (15): negatives -> lamp, lantern, candle
- gravy_boat (10): negatives -> bowl, pot, pitcher_(vessel_for_liquid)
- quiche (10): negatives -> pie, pizza, omelet

Reserve/secondary targets (valid but likely less globally rare):

- parakeet (11): negatives -> bird, pigeon, hummingbird
- bean_curd (24): negatives -> hummus, sour_cream, mashed_potato
- plow_(farm_equipment) (10): negatives -> power_shovel, shovel
- chap (10): negatives -> trousers, jumpsuit
- parchment (10): negatives -> booklet, map, notebook
