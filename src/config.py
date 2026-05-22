from pathlib import Path

import torch


# =========================================================
# Project Paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_RECORDS_DIR = DATA_DIR / "processed" / "records"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"

FINAL_SPLIT_CSV = PROCESSED_RECORDS_DIR / "final_train_val_test_split.csv"
AUDITED_RECORDS_CSV = PROCESSED_RECORDS_DIR / "audited_classification_records.csv"


# =========================================================
# Create Output Directories
# =========================================================

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# Fixed Class Mapping
# =========================================================

TARGET_CLASSES = ["plastic", "foam", "metal", "other_debris"]

CLASS_TO_IDX = {
    "plastic": 0,
    "foam": 1,
    "metal": 2,
    "other_debris": 3,
}

IDX_TO_CLASS = {
    0: "plastic",
    1: "foam",
    2: "metal",
    3: "other_debris",
}


# =========================================================
# Dataset Pipeline Settings
# =========================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 0
RANDOM_SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =========================================================
# Device
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")