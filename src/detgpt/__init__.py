from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# data paths
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# models paths
MODELS_DIR = BASE_DIR / "models"

# outputs/logs paths
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# configs paths
CONFIGS_DIR = BASE_DIR / "configs"
