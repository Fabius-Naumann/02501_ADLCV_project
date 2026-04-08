## Documentation

# DetGPT Technical Documentation

## Overview
DetGPT explores how off-the-shelf VLMs can perform few-shot object detection when shown visual examples of a target object without any fine-tuning. Traditional text-only prompts are often too ambiguous to capture precise visual concepts; ICL allows the model to learn from the visual context provided in the prompt.

## Task Roadmap
* **Task 1: Zero-shot Performance**: Benchmarking VLMs (1-8B parameters) against SOTA detectors like **Grounding DINO** and **YOLO-World**.
* **Task 2: In-Context Design**: Evaluating different layout strategies such as side-by-side images or cropped exemplars in 1-shot and 5-shot settings.
* **Task 3: VLM + Detector Fusion**: Combining the visual understanding of VLMs with the precise localization of specialized detectors.


# Data Infrastructure

## Dataset Design: `Task1DetectionDataset`
The dataset implementation in `data.py` is designed for **Task 1: Zero-shot Performance Evaluation**. It bridges the `LvisAPI` output with the training/inference loops.

### Coordinate Transformation
To support the spatial reasoning capabilities of VLMs, we transform raw LVIS bounding boxes into center-normalized coordinates:
* **Input**: $[x_{min}, y_{min}, width, height]$
* **Transformation**:
  $$x_{center} = x_{min} + \frac{width}{2}$$
  $$y_{center} = y_{min} + \frac{height}{2}$$
* **Output**: The `_extract_bbox_xcycwh` method returns these as a unified tensor for the detector.

### Manifest-Backed Architecture
By using a local manifest (`lvis_v1_train_manifest.json`), we achieve:
1. **O(1) Access**: Models don't need to parse the entire LVIS annotation file (1GB+) at runtime.
2. **Persistence**: The manifest stores the `local_path`, ensuring that the dataset is portable across different HPC environments as long as the relative paths are maintained.
3. **Reproducibility**: Filtering logic is separated from model logic; the model always sees the exact same samples defined in the manifest.

### Batch Processing
Because object detection involves variable-length targets, the `task1_collate_fn` is utilized. It prevents the default `DataLoader` from attempting to stack target dictionaries of different sizes, which would otherwise result in a `RuntimeError`.

## Evaluation Scripts

Two evaluation entrypoints are available:

1. `detgpt.evaluate` for model inference over the prepared Task 1 dataset.
2. `detgpt.evaluate_files` for metrics computed from JSON prediction and ground-truth files.

### Inference Evaluation (`evaluate.py`)

Run Grounding DINO baseline evaluation:

```bash
uv run python -m detgpt.evaluate \
  --detector-backend grounding_dino \
  --split train \
  --limit 20 \
  --save-results \
  --save-viz
```

Run Qwen-VLM evaluation with deterministic decoding and debug trace:

```bash
uv run python -m detgpt.evaluate \
  --detector-backend qwen_vlm \
  --model-id Qwen/Qwen3.5-2B \
  --split train \
  --limit 20 \
  --qwen-max-detections-per-category 5 \
  --qwen-temperature 0.0 \
  --qwen-debug-dump \
  --save-viz
```

Each run writes outputs to `outputs/task1_results/run_<timestamp>/`.
When Qwen debug dumping is enabled, `qwen_debug_trace.jsonl` is saved in the same run directory.

### File-Based Metrics Evaluation (`evaluate_files.py`)

Use this path when predictions are already stored as JSON files (to be refined - TODO: @Alexandra):

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

Expected record format:

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

`boxes` must be in `cxcywh` format for both predictions and ground truth.
