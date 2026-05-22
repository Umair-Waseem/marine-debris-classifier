from pathlib import Path
import json
import random

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    PROJECT_ROOT,
    PROCESSED_RECORDS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    DEVICE,
)

from transforms import (
    get_train_transforms,
    get_eval_transforms,
    denormalize_tensor_image,
)

from multilabel_dataset import MarineDebrisMultiLabelDataset


# =========================================================
# Multi-Label Configuration
# =========================================================

MULTILABEL_TRAIN_VAL_CSV = PROCESSED_RECORDS_DIR / "multilabel_train_val_records.csv"

TARGET_CLASSES = ["plastic", "foam", "metal", "other_debris"]

TARGET_COLUMNS = [
    "multilabel_plastic",
    "multilabel_foam",
    "multilabel_metal",
    "multilabel_other_debris",
]


# =========================================================
# Output Paths
# =========================================================

SUMMARY_JSON = REPORTS_DIR / "multilabel_dataset_pipeline_summary.json"
REPORT_MD = REPORTS_DIR / "multilabel_dataset_pipeline_report.md"
TRAIN_BATCH_GRID = PLOTS_DIR / "multilabel_train_batch_grid.png"


# =========================================================
# Helper Functions
# =========================================================

def set_seed(seed: int):
    """Set random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, data: dict):
    """Save JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def targets_are_binary(targets: torch.Tensor) -> bool:
    """Check whether target tensor contains only 0 or 1."""
    unique_values = torch.unique(targets.detach().cpu())

    for value in unique_values:
        item = float(value.item())

        if item not in [0.0, 1.0]:
            return False

    return True


def all_classes_have_positive_counts(positive_counts: dict) -> bool:
    """Check all target classes have positive samples."""
    return all(
        int(positive_counts.get(class_name, 0)) > 0
        for class_name in TARGET_CLASSES
    )


def check_batch(batch, split_name: str):
    """
    Check one DataLoader batch.

    Expected:
    - image shape: [16, 3, 224, 224]
    - target shape: [16, 4]
    - target dtype: float32
    - target values: 0 or 1
    """
    images, targets, metadata = batch

    image_shape = [int(x) for x in images.shape]
    target_shape = [int(x) for x in targets.shape]

    image_shape_ok = (
        images.ndim == 4
        and images.shape[0] == BATCH_SIZE
        and images.shape[1] == 3
        and images.shape[2] == IMAGE_SIZE
        and images.shape[3] == IMAGE_SIZE
    )

    target_shape_ok = (
        targets.ndim == 2
        and targets.shape[0] == images.shape[0]
        and targets.shape[1] == len(TARGET_CLASSES)
    )

    target_dtype_ok = targets.dtype == torch.float32
    target_binary_ok = targets_are_binary(targets)

    batch_check_passed = bool(
        image_shape_ok
        and target_shape_ok
        and target_dtype_ok
        and target_binary_ok
    )

    example_targets = [
        [float(value) for value in row]
        for row in targets.detach().cpu()[:5].tolist()
    ]

    return {
        "split": split_name,
        "image_shape": image_shape,
        "target_shape": target_shape,
        "image_dtype": str(images.dtype),
        "target_dtype": str(targets.dtype),
        "target_min": float(targets.min().item()),
        "target_max": float(targets.max().item()),
        "example_target_vectors": example_targets,
        "image_shape_ok": bool(image_shape_ok),
        "target_shape_ok": bool(target_shape_ok),
        "target_dtype_ok": bool(target_dtype_ok),
        "target_binary_ok": bool(target_binary_ok),
        "batch_check_passed": bool(batch_check_passed),
    }


def metadata_to_label_list(metadata_batch, index: int):
    """
    Extract positive-label title from collated metadata.
    """
    if "multilabel_label_names" not in metadata_batch:
        return ""

    value = metadata_batch["multilabel_label_names"]

    if isinstance(value, list):
        return str(value[index])

    if isinstance(value, tuple):
        return str(value[index])

    return str(value)


def save_multilabel_batch_grid(images, metadata_batch, output_path: Path, max_images: int = 16):
    """
    Save a visual grid of one multi-label training batch.

    Images are denormalized for plotting only.
    No dataset images are modified.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = images.detach().cpu()

    n_images = min(max_images, images.shape[0])
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if hasattr(axes, "flatten"):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for ax in axes_flat:
        ax.axis("off")

    for i in range(n_images):
        image = denormalize_tensor_image(images[i])
        image_np = image.permute(1, 2, 0).numpy()

        label_text = metadata_to_label_list(metadata_batch, i)

        if len(label_text) > 35:
            label_text = label_text[:35] + "..."

        axes_flat[i].imshow(image_np)
        axes_flat[i].set_title(label_text, fontsize=8)
        axes_flat[i].axis("off")

    fig.suptitle("Multi-Label Train Batch Grid", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def markdown_table_from_dict(title: str, data: dict) -> str:
    """Create Markdown table from dictionary."""
    lines = [
        f"### {title}",
        "",
        "| Item | Value |",
        "|---|---:|",
    ]

    for key, value in data.items():
        lines.append(f"| {key} | {value} |")

    return "\n".join(lines)


# =========================================================
# Main
# =========================================================

def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("MULTI-LABEL DATASET PIPELINE CHECK STARTED")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Device:", DEVICE)
    print("Records CSV:", MULTILABEL_TRAIN_VAL_CSV)
    print("Target classes:", TARGET_CLASSES)
    print("Target columns:", TARGET_COLUMNS)

    if not MULTILABEL_TRAIN_VAL_CSV.exists():
        raise FileNotFoundError(
            f"Multi-label train/validation CSV not found: {MULTILABEL_TRAIN_VAL_CSV}\n"
            "Run src/create_multilabel_records.py first."
        )

    # -----------------------------------------------------
    # Create Datasets
    # -----------------------------------------------------

    train_dataset = MarineDebrisMultiLabelDataset(
        records_csv=MULTILABEL_TRAIN_VAL_CSV,
        split="train",
        transform=get_train_transforms(),
        target_classes=TARGET_CLASSES,
        target_columns=TARGET_COLUMNS,
        require_readable=True,
        return_metadata=True,
    )

    validation_dataset = MarineDebrisMultiLabelDataset(
        records_csv=MULTILABEL_TRAIN_VAL_CSV,
        split="validation",
        transform=get_eval_transforms(),
        target_classes=TARGET_CLASSES,
        target_columns=TARGET_COLUMNS,
        require_readable=True,
        return_metadata=True,
    )

    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
    }

    print("\nDataset lengths:")
    for split_name, dataset in datasets.items():
        print(f"- {split_name}: {len(dataset)}")

    print("\nPositive class counts and label cardinality:")
    class_presence_checks = {}

    for split_name, dataset in datasets.items():
        positive_counts = dataset.get_positive_counts()
        label_cardinality = dataset.get_label_cardinality()
        class_presence = all_classes_have_positive_counts(positive_counts)
        class_presence_checks[split_name] = class_presence

        print(f"\n{split_name}:")
        for class_name, count in positive_counts.items():
            print(f"  {class_name}: {count}")

        print("  label cardinality:", label_cardinality)
        print("  all classes present:", class_presence)

    # -----------------------------------------------------
    # Create DataLoaders
    # -----------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    loaders = {
        "train": train_loader,
        "validation": validation_loader,
    }

    # -----------------------------------------------------
    # Batch Checks
    # -----------------------------------------------------

    batch_checks = {}

    print("\nBatch checks:")

    for split_name, loader in loaders.items():
        batch = next(iter(loader))
        batch_info = check_batch(batch, split_name)
        batch_checks[split_name] = batch_info

        print(f"\n{split_name} batch:")
        print("  image shape:", batch_info["image_shape"])
        print("  target shape:", batch_info["target_shape"])
        print("  image dtype:", batch_info["image_dtype"])
        print("  target dtype:", batch_info["target_dtype"])
        print("  target min:", batch_info["target_min"])
        print("  target max:", batch_info["target_max"])
        print("  example target vectors:", batch_info["example_target_vectors"])
        print("  batch check passed:", batch_info["batch_check_passed"])

    # Save visual grid from train batch.
    train_images, train_targets, train_metadata = next(iter(train_loader))

    save_multilabel_batch_grid(
        images=train_images,
        metadata_batch=train_metadata,
        output_path=TRAIN_BATCH_GRID,
        max_images=16,
    )

    # -----------------------------------------------------
    # Final Pass/Fail Logic
    # -----------------------------------------------------

    all_datasets_loaded = all(len(dataset) > 0 for dataset in datasets.values())

    all_classes_present = all(class_presence_checks.values())

    all_batches_passed = all(
        batch_info["batch_check_passed"]
        for batch_info in batch_checks.values()
    )

    expected_shapes_ok = all(
        batch_info["image_shape"] == [BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE]
        and batch_info["target_shape"] == [BATCH_SIZE, len(TARGET_CLASSES)]
        for batch_info in batch_checks.values()
    )

    targets_binary_ok = all(
        batch_info["target_binary_ok"]
        for batch_info in batch_checks.values()
    )

    pipeline_check_passed = bool(
        all_datasets_loaded
        and all_classes_present
        and all_batches_passed
        and expected_shapes_ok
        and targets_binary_ok
    )

    if pipeline_check_passed:
        recommended_next_step = (
            "Proceed to multi-label model training using ResNet18 with BCEWithLogitsLoss. "
            "Do not overwrite the official single-label model."
        )
    else:
        recommended_next_step = (
            "Fix the multi-label dataset pipeline before training."
        )

    # -----------------------------------------------------
    # Save Summary JSON
    # -----------------------------------------------------

    dataset_summaries = {
        split_name: dataset.get_summary()
        for split_name, dataset in datasets.items()
    }

    summary = {
        "project_root": str(PROJECT_ROOT),
        "device": str(DEVICE),
        "records_csv": str(MULTILABEL_TRAIN_VAL_CSV),
        "target_classes": TARGET_CLASSES,
        "target_columns": TARGET_COLUMNS,
        "batch_size": int(BATCH_SIZE),
        "image_size": int(IMAGE_SIZE),
        "num_workers": int(NUM_WORKERS),
        "dataset_summaries": dataset_summaries,
        "class_presence_checks": {
            split_name: bool(value)
            for split_name, value in class_presence_checks.items()
        },
        "batch_checks": batch_checks,
        "all_datasets_loaded": bool(all_datasets_loaded),
        "all_classes_present": bool(all_classes_present),
        "all_batches_passed": bool(all_batches_passed),
        "expected_shapes_ok": bool(expected_shapes_ok),
        "targets_binary_ok": bool(targets_binary_ok),
        "pipeline_check_passed": bool(pipeline_check_passed),
        "training_performed": False,
        "test_split_loaded": False,
        "official_single_label_model_untouched": True,
        "output_files": {
            "summary_json": str(SUMMARY_JSON),
            "report_md": str(REPORT_MD),
            "train_batch_grid": str(TRAIN_BATCH_GRID),
        },
        "recommended_next_step": recommended_next_step,
    }

    save_json(SUMMARY_JSON, summary)

    # -----------------------------------------------------
    # Save Markdown Report
    # -----------------------------------------------------

    report_lines = []

    report_lines.append("# Multi-Label Dataset Pipeline Report")
    report_lines.append("")
    report_lines.append("## 1. Purpose")
    report_lines.append("")
    report_lines.append(
        "This report verifies the PyTorch Dataset and DataLoader pipeline for "
        "the experimental multi-label branch. No model training is performed."
    )

    report_lines.append("")
    report_lines.append("## 2. Important Notes")
    report_lines.append("")
    report_lines.append("- The official single-label model is untouched.")
    report_lines.append("- The test split is not loaded.")
    report_lines.append("- No training is performed.")
    report_lines.append("- No raw images are modified.")
    report_lines.append("- Multi-label target order is `[plastic, foam, metal, other_debris]`.")

    report_lines.append("")
    report_lines.append("## 3. Dataset Lengths")
    report_lines.append("")
    report_lines.append("| Split | Length | Label Cardinality |")
    report_lines.append("|---|---:|---:|")

    for split_name, dataset in datasets.items():
        report_lines.append(
            f"| {split_name} | {len(dataset)} | {dataset.get_label_cardinality():.6f} |"
        )

    report_lines.append("")
    report_lines.append("## 4. Positive Class Counts")
    report_lines.append("")

    for split_name, dataset in datasets.items():
        report_lines.append(markdown_table_from_dict(split_name, dataset.get_positive_counts()))
        report_lines.append("")

    report_lines.append("")
    report_lines.append("## 5. Batch Checks")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(batch_checks, indent=4))
    report_lines.append("```")

    report_lines.append("")
    report_lines.append("## 6. Final Decision")
    report_lines.append("")
    report_lines.append(f"- Pipeline check passed: `{pipeline_check_passed}`")
    report_lines.append(f"- Test split loaded: `False`")
    report_lines.append(f"- Training performed: `False`")
    report_lines.append("")
    report_lines.append(recommended_next_step)

    report_lines.append("")
    report_lines.append("## 7. Output Files")
    report_lines.append("")
    for name, path in summary["output_files"].items():
        report_lines.append(f"- {name}: `{path}`")

    REPORT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    # -----------------------------------------------------
    # Console Summary
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("MULTI-LABEL DATASET PIPELINE SUMMARY")
    print("=" * 80)

    print("\nDataset lengths:")
    for split_name, dataset in datasets.items():
        print(f"- {split_name}: {len(dataset)}")

    print("\nPositive counts:")
    for split_name, dataset in datasets.items():
        print(f"{split_name}: {dataset.get_positive_counts()}")

    print("\nLabel cardinality:")
    for split_name, dataset in datasets.items():
        print(f"{split_name}: {dataset.get_label_cardinality():.6f}")

    print("\nBatch checks:")
    for split_name, info in batch_checks.items():
        print(f"{split_name}: {info['batch_check_passed']}")

    print("\nFinal decision:")
    print("Pipeline check passed:", pipeline_check_passed)
    print("Test split loaded:", False)
    print("Training performed:", False)
    print(recommended_next_step)

    print("\nSaved output files:")
    print("-", SUMMARY_JSON)
    print("-", REPORT_MD)
    print("-", TRAIN_BATCH_GRID)

    print("\nMULTI-LABEL DATASET PIPELINE CHECK COMPLETED.")


if __name__ == "__main__":
    main()