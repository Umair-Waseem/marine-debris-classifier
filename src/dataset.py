from pathlib import Path
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError

from config import PROJECT_ROOT, TARGET_CLASSES, CLASS_TO_IDX


# =========================================================
# Helper Functions
# =========================================================

def normalize_text(value) -> str:
    """Convert a value to a clean string."""
    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_label(value) -> str:
    """Normalize class label text."""
    return normalize_text(value).lower()


def to_bool(value) -> bool:
    """Convert common boolean-like values to bool."""
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y"}:
        return True

    if text in {"false", "0", "no", "n", ""}:
        return False

    return False


def resolve_path_from_row(row: pd.Series) -> Path:
    """
    Resolve image path safely.

    Priority:
    1. audit_resolved_image_path
    2. image_path
    3. image_path_relative
    """
    candidate_columns = [
        "audit_resolved_image_path",
        "image_path",
        "image_path_relative",
    ]

    fallback_path = None

    for column in candidate_columns:
        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        text = str(value).strip()

        if not text:
            continue

        path = Path(text)

        if path.is_absolute():
            candidate = path
        else:
            candidate = PROJECT_ROOT / path

        if fallback_path is None:
            fallback_path = candidate

        if candidate.exists():
            return candidate

    if fallback_path is not None:
        return fallback_path

    return PROJECT_ROOT / "missing_image_path"


# =========================================================
# Dataset Class
# =========================================================

class MarineDebrisDataset(Dataset):
    """
    PyTorch Dataset for the Marine Debris Classifier project.

    This dataset reads image paths and labels from:
        data/processed/records/final_train_val_test_split.csv

    It does not copy, resize, or modify raw images on disk.
    """

    VALID_SPLITS = {"train", "validation", "test"}

    def __init__(
        self,
        records_csv,
        split,
        transform=None,
        target_classes=None,
        class_to_idx=None,
        require_readable=True,
        include_ambiguous=True,
        return_metadata=False,
    ):
        self.records_csv = Path(records_csv)
        self.split = str(split).strip().lower()
        self.transform = transform
        self.target_classes = (
            list(target_classes)
            if target_classes is not None
            else list(TARGET_CLASSES)
        )
        self.class_to_idx = (
            dict(class_to_idx)
            if class_to_idx is not None
            else dict(CLASS_TO_IDX)
        )
        self.idx_to_class = {
            idx: class_name
            for class_name, idx in self.class_to_idx.items()
        }
        self.require_readable = bool(require_readable)
        self.include_ambiguous = bool(include_ambiguous)
        self.return_metadata = bool(return_metadata)

        self._validate_inputs()
        self.records = self._load_and_filter_records()

        self.class_counts = Counter(self.records["selected_label_clean"].tolist())
        self.num_classes = len(self.target_classes)

        if len(self.records) == 0:
            raise ValueError(
                "Dataset is empty after filtering.\n"
                f"Split: {self.split}\n"
                f"include_ambiguous: {self.include_ambiguous}\n"
                f"records_csv: {self.records_csv}"
            )

    def _validate_inputs(self):
        """Validate constructor inputs."""
        if not self.records_csv.exists():
            raise FileNotFoundError(f"Records CSV not found: {self.records_csv}")

        if self.split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {self.split}. "
                "Expected one of: train, validation, test."
            )

        missing_classes = [
            class_name
            for class_name in self.target_classes
            if class_name not in self.class_to_idx
        ]

        if missing_classes:
            raise ValueError(
                "class_to_idx is missing target classes: "
                + ", ".join(missing_classes)
            )

    def _load_and_filter_records(self) -> pd.DataFrame:
        """Load final split CSV and apply safe filters."""
        df = pd.read_csv(self.records_csv)

        required_columns = [
            "final_split",
            "filename",
            "selected_label",
            "usable_for_multiclass",
            "caution_flag",
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "Records CSV is missing required columns:\n"
                + "\n".join(f"- {col}" for col in missing_columns)
            )

        df = df.copy()

        df["final_split_clean"] = df["final_split"].apply(normalize_text).str.lower()
        df["selected_label_clean"] = df["selected_label"].apply(normalize_label)
        df["usable_for_multiclass_bool"] = df["usable_for_multiclass"].apply(to_bool)

        if "audit_image_exists" in df.columns:
            df["audit_image_exists_bool"] = df["audit_image_exists"].apply(to_bool)
        else:
            df["audit_image_exists_bool"] = True

        if "audit_image_readable" in df.columns:
            df["audit_image_readable_bool"] = df["audit_image_readable"].apply(to_bool)
        else:
            df["audit_image_readable_bool"] = True

        if "is_ambiguous" in df.columns:
            df["is_ambiguous_bool"] = df["is_ambiguous"].apply(to_bool)
        else:
            df["is_ambiguous_bool"] = (
                df["caution_flag"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("ambiguous_multilabel")
            )

        # Filter by final split.
        df = df[df["final_split_clean"] == self.split].copy()

        # Use only multiclass-usable records.
        df = df[df["usable_for_multiclass_bool"] == True].copy()

        # Keep only fixed target classes.
        df = df[df["selected_label_clean"].isin(self.target_classes)].copy()

        # Use only audited existing images if audit column is available.
        if "audit_image_exists" in df.columns:
            df = df[df["audit_image_exists_bool"] == True].copy()

        # Use only readable images if required and audit column is available.
        if self.require_readable and "audit_image_readable" in df.columns:
            df = df[df["audit_image_readable_bool"] == True].copy()

        # Optional: exclude ambiguous records.
        if not self.include_ambiguous:
            df = df[
                df["caution_flag"]
                .astype(str)
                .str.strip()
                .str.lower()
                .ne("ambiguous_multilabel")
            ].copy()

        # Resolve image paths.
        resolved_paths = []
        resolved_exists = []

        for _, row in df.iterrows():
            image_path = resolve_path_from_row(row)
            resolved_paths.append(str(image_path))
            resolved_exists.append(image_path.exists())

        df["resolved_image_path"] = resolved_paths
        df["resolved_image_exists"] = resolved_exists

        df = df[df["resolved_image_exists"] == True].copy()
        df = df.reset_index(drop=True)

        return df

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]

        image_path = Path(row["resolved_image_path"])
        label_name = row["selected_label_clean"]

        if label_name not in self.class_to_idx:
            raise ValueError(
                f"Label '{label_name}' is not present in class_to_idx."
            )

        label_idx = self.class_to_idx[label_name]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                if self.transform is not None:
                    image = self.transform(image)

        except (UnidentifiedImageError, OSError, Exception) as exc:
            raise RuntimeError(
                f"Could not open image: {image_path}\nError: {exc}"
            ) from exc

        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        if not self.return_metadata:
            return image, label_tensor

        metadata = {
            "filename": str(row.get("filename", "")),
            "image_path": str(image_path),
            "selected_label": str(label_name),
            "final_split": str(row.get("final_split_clean", self.split)),
            "caution_flag": str(row.get("caution_flag", "")),
            "is_ambiguous": bool(row.get("is_ambiguous_bool", False)),
        }

        return image, label_tensor, metadata

    def get_class_distribution(self):
        """Return class distribution as a dictionary."""
        return {
            class_name: int(self.class_counts.get(class_name, 0))
            for class_name in self.target_classes
        }

    def get_summary(self):
        """Return dataset summary as a dictionary."""
        return {
            "records_csv": str(self.records_csv),
            "split": self.split,
            "num_samples": int(len(self.records)),
            "num_classes": int(self.num_classes),
            "target_classes": list(self.target_classes),
            "class_to_idx": dict(self.class_to_idx),
            "idx_to_class": {
                str(idx): class_name
                for idx, class_name in self.idx_to_class.items()
            },
            "class_distribution": self.get_class_distribution(),
            "include_ambiguous": bool(self.include_ambiguous),
            "require_readable": bool(self.require_readable),
        }