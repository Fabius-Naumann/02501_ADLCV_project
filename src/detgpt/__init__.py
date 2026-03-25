from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# data paths
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# models paths
MODELS_DIR = BASE_DIR / "models"

# outputs/logs paths
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# configs paths
CONFIGS_DIR = BASE_DIR / "configs"


def init_data_dirs() -> None:
    """
    Initialize data directories used by detgpt.
    This function creates the RAW_DIR and PROCESSED_DIR directories if they do
    not already exist. It should be called explicitly by CLI/setup code in
    environments where these directories are needed.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
