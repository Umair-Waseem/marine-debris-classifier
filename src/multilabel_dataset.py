from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError

from config import PROJECT_ROOT


# =========================================================
# Multi-Label Class Configuration
# =========================================================

DEFAULT_TARGET_CLASSES = ["plastic", "foam", "metal", "other_debris"]

DEFAULT_TARGET_COLUMNS = [
    "multilabel_plastic",
    "multilabel_foam",
    "multilabel_metal",
    "multilabel_other_debris",
]


# =========================================================
# Helper Functions
# =========================================================

def normalize_text(value) -> str:
    """Convert value to clean string."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_label(value) -> str:
    """Normalize label text."""
    return normalize_text(value).lower()


def to_bool(value) -> bool:
    """Convert common boolean-like values to bool."""
    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    text = str(value).strip().lower()

    return text in {"true", "1", "yes", "y"}


def resolve_image_path_from_row(row: pd.Series) -> Path:
    """
    Resolve image path safely.

    Priority:
    1. audit_resolved_image_path
    2. resolved_image_path
    3. image_path
    4. image_path_relative
    """
    candidate_columns = [
        "audit_resolved_image_path",
        "resolved_image_path",
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

        if text == "":
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


def validate_binary_target_values(df: pd.DataFrame, target_columns: list[str]):
    """Verify that target columns contain only 0 or 1."""
    errors = []

    for column in target_columns:
        if column not in df.columns:
            errors.append(f"Missing target column: {column}")
            continue

        values = pd.to_numeric(df[column], errors="coerce")
        unique_values = sorted(values.dropna().unique().tolist())

        invalid_values = [
            value for value in unique_values
            if value not in [0, 1]
        ]

        if invalid_values:
            errors.append(
                f"Target column '{column}' contains invalid values: {invalid_values}"
            )

    if errors:
        raise ValueError(
            "Invalid multi-label target columns:\n"
            + "\n".join(f"- {error}" for error in errors)
        )


# =========================================================
# Multi-Label Dataset Class
# =========================================================

class MarineDebrisMultiLabelDataset(Dataset):
    """
    PyTorch Dataset for multi-label marine debris classification.

    Each sample returns:
        image tensor
        multi-hot target tensor with shape [4], dtype float32

    Target order:
        [plastic, foam, metal, other_debris]
    """

    VALID_SPLITS = {"train", "validation"}

    def __init__(
        self,
        records_csv,
        split,
        transform=None,
        target_classes=None,
        target_columns=None,
        require_readable=True,
        return_metadata=False,
    ):
        self.records_csv = Path(records_csv)
        self.split = str(split).strip().lower()
        self.transform = transform
        self.target_classes = (
            list(target_classes)
            if target_classes is not None
            else list(DEFAULT_TARGET_CLASSES)
        )
        self.target_columns = (
            list(target_columns)
            if target_columns is not None
            else list(DEFAULT_TARGET_COLUMNS)
        )
        self.require_readable = bool(require_readable)
        self.return_metadata = bool(return_metadata)
        self.num_classes = len(self.target_classes)

        self._validate_inputs()
        self.records = self._load_and_filter_records()

        if len(self.records) == 0:
            raise ValueError(
                "Dataset is empty after filtering.\n"
                f"records_csv: {self.records_csv}\n"
                f"split: {self.split}"
            )

    def _validate_inputs(self):
        """Validate constructor inputs."""
        if not self.records_csv.exists():
            raise FileNotFoundError(f"Records CSV not found: {self.records_csv}")

        if self.split not in self.VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {self.split}. "
                "Expected one of: train, validation."
            )

        if len(self.target_classes) != len(self.target_columns):
            raise ValueError(
                "target_classes and target_columns must have the same length."
            )

    def _load_and_filter_records(self) -> pd.DataFrame:
        """Load records CSV and apply multi-label filters."""
        df = pd.read_csv(self.records_csv)

        required_columns = [
            "filename",
            "final_split",
            "usable_for_multiclass",
            "selected_label",
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

        missing_target_columns = [
            col for col in self.target_columns
            if col not in df.columns
        ]

        if missing_target_columns:
            raise ValueError(
                "Records CSV is missing target columns:\n"
                + "\n".join(f"- {col}" for col in missing_target_columns)
            )

        df = df.copy()

        df["filename"] = df["filename"].apply(normalize_text)
        df["final_split"] = df["final_split"].apply(normalize_text).str.lower()
        df["selected_label"] = df["selected_label"].apply(normalize_label)
        df["caution_flag"] = df["caution_flag"].apply(normalize_text).str.lower()
        df["usable_for_multiclass"] = df["usable_for_multiclass"].apply(to_bool)

        if "is_ambiguous" in df.columns:
            df["is_ambiguous"] = df["is_ambiguous"].apply(to_bool)
        else:
            df["is_ambiguous"] = df["caution_flag"].eq("ambiguous_multilabel")

        # Filter split and usable records.
        df = df[df["final_split"] == self.split].copy()
        df = df[df["usable_for_multiclass"] == True].copy()

        # Convert targets to int and validate.
        for column in self.target_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0).astype(int)

        validate_binary_target_values(df, self.target_columns)

        # Require at least one positive label.
        df["multilabel_target_sum"] = df[self.target_columns].sum(axis=1)
        df = df[df["multilabel_target_sum"] > 0].copy()

        # Optional image audit filters.
        if "audit_image_exists" in df.columns:
            df["audit_image_exists"] = df["audit_image_exists"].apply(to_bool)
            df = df[df["audit_image_exists"] == True].copy()

        if self.require_readable and "audit_image_readable" in df.columns:
            df["audit_image_readable"] = df["audit_image_readable"].apply(to_bool)
            df = df[df["audit_image_readable"] == True].copy()

        # Resolve image paths.
        resolved_paths = []
        resolved_exists = []

        for _, row in df.iterrows():
            image_path = resolve_image_path_from_row(row)
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

        target_values = [
            float(row[column])
            for column in self.target_columns
        ]

        target_tensor = torch.tensor(target_values, dtype=torch.float32)

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")

                if self.transform is not None:
                    image = self.transform(image)

        except (UnidentifiedImageError, OSError, Exception) as exc:
            raise RuntimeError(
                f"Could not open image: {image_path}\nError: {exc}"
            ) from exc

        if not self.return_metadata:
            return image, target_tensor

        positive_labels = [
            class_name
            for class_name, value in zip(self.target_classes, target_values)
            if int(value) == 1
        ]

        metadata = {
            "filename": str(row.get("filename", "")),
            "image_path": str(image_path),
            "final_split": str(row.get("final_split", self.split)),
            "selected_label": str(row.get("selected_label", "")),
            "multilabel_label_names": ", ".join(positive_labels),
            "multilabel_num_positive_classes": int(sum(target_values)),
            "caution_flag": str(row.get("caution_flag", "")),
            "is_ambiguous": bool(row.get("is_ambiguous", False)),
        }

        return image, target_tensor, metadata

    def get_positive_counts(self):
        """Return positive counts for each target class."""
        counts = {}

        for class_name, column in zip(self.target_classes, self.target_columns):
            counts[class_name] = int(self.records[column].sum())

        return counts

    def get_label_cardinality(self):
        """
        Return average number of positive labels per image.
        """
        if len(self.records) == 0:
            return 0.0

        total_positive_labels = 0

        for column in self.target_columns:
            total_positive_labels += int(self.records[column].sum())

        return float(total_positive_labels / len(self.records))

    def get_summary(self):
        """Return dataset summary dictionary."""
        return {
            "records_csv": str(self.records_csv),
            "split": self.split,
            "num_samples": int(len(self.records)),
            "target_classes": list(self.target_classes),
            "target_columns": list(self.target_columns),
            "num_classes": int(self.num_classes),
            "positive_counts": self.get_positive_counts(),
            "label_cardinality": self.get_label_cardinality(),
            "require_readable": bool(self.require_readable),
            "return_metadata": bool(self.return_metadata),
        }
