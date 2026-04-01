````markdown

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

````
