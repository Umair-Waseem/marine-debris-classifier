import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    PROJECT_ROOT,
    FINAL_SPLIT_CSV,
    REPORTS_DIR,
    PLOTS_DIR,
    TARGET_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
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

from dataset import MarineDebrisDataset


# =========================================================
# Output Paths
# =========================================================

CLASS_MAPPING_JSON_PATH = REPORTS_DIR / "class_mapping.json"
PIPELINE_SUMMARY_JSON_PATH = REPORTS_DIR / "dataset_pipeline_summary.json"
PIPELINE_REPORT_PATH = REPORTS_DIR / "dataset_pipeline_report.md"
TRANSFORMED_GRID_PATH = PLOTS_DIR / "transformed_train_batch_grid.png"


# =========================================================
# Helper Functions
# =========================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, data: dict):
    """Save dictionary as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def all_classes_present(class_distribution: dict) -> bool:
    """Check whether all target classes appear in a split."""
    return all(
        int(class_distribution.get(class_name, 0)) > 0
        for class_name in TARGET_CLASSES
    )


def check_batch(batch, split_name: str):
    """
    Check one DataLoader batch.

    Expected:
    - image tensor shape: [16, 3, 224, 224]
    - label tensor shape: [16]
    - label indices from 0 to 3
    """
    images, labels = batch

    batch_size = images.shape[0]

    shape_ok = (
        images.ndim == 4
        and batch_size == BATCH_SIZE
        and images.shape[1] == 3
        and images.shape[2] == IMAGE_SIZE
        and images.shape[3] == IMAGE_SIZE
    )

    labels_shape_ok = (
        labels.ndim == 1
        and labels.shape[0] == batch_size
    )

    labels_range_ok = bool(
        labels.min().item() >= 0
        and labels.max().item() < len(TARGET_CLASSES)
    )

    batch_check_passed = bool(
        shape_ok
        and labels_shape_ok
        and labels_range_ok
    )

    return {
        "split": split_name,
        "image_shape": [int(x) for x in images.shape],
        "label_shape": [int(x) for x in labels.shape],
        "image_dtype": str(images.dtype),
        "label_dtype": str(labels.dtype),
        "image_min": float(images.min().item()),
        "image_max": float(images.max().item()),
        "labels": [int(x) for x in labels.detach().cpu().tolist()],
        "shape_ok": bool(shape_ok),
        "labels_shape_ok": bool(labels_shape_ok),
        "labels_range_ok": bool(labels_range_ok),
        "batch_check_passed": bool(batch_check_passed),
    }


def save_transformed_batch_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
    output_path: Path,
    max_images: int = 16,
):
    """
    Save transformed training batch grid.

    Images are denormalized only for visualization.
    No dataset images are modified or saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images = images.detach().cpu()
    labels = labels.detach().cpu()

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

        label_idx = int(labels[i].item())
        class_name = IDX_TO_CLASS[label_idx]

        axes_flat[i].imshow(image_np)
        axes_flat[i].set_title(class_name, fontsize=10)
        axes_flat[i].axis("off")

    fig.suptitle("Transformed Train Batch Grid", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def markdown_table_from_dict(title: str, data: dict) -> str:
    """Create a small markdown table from a dictionary."""
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
# Main Check Function
# =========================================================

def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("DATASET PIPELINE CHECK STARTED")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Device:", DEVICE)
    print("Final split CSV:", FINAL_SPLIT_CSV)
    print("Class mapping:", CLASS_TO_IDX)

    if not FINAL_SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Final split CSV not found: {FINAL_SPLIT_CSV}\n"
            "Run the dataset audit and split strategy script before this check."
        )

    # -----------------------------------------------------
    # Create Datasets
    # -----------------------------------------------------

    train_dataset = MarineDebrisDataset(
        records_csv=FINAL_SPLIT_CSV,
        split="train",
        transform=get_train_transforms(),
        target_classes=TARGET_CLASSES,
        class_to_idx=CLASS_TO_IDX,
        require_readable=True,
        include_ambiguous=True,
        return_metadata=False,
    )

    validation_dataset = MarineDebrisDataset(
        records_csv=FINAL_SPLIT_CSV,
        split="validation",
        transform=get_eval_transforms(),
        target_classes=TARGET_CLASSES,
        class_to_idx=CLASS_TO_IDX,
        require_readable=True,
        include_ambiguous=True,
        return_metadata=False,
    )

    test_dataset = MarineDebrisDataset(
        records_csv=FINAL_SPLIT_CSV,
        split="test",
        transform=get_eval_transforms(),
        target_classes=TARGET_CLASSES,
        class_to_idx=CLASS_TO_IDX,
        require_readable=True,
        include_ambiguous=True,
        return_metadata=False,
    )

    datasets = {
        "train": train_dataset,
        "validation": validation_dataset,
        "test": test_dataset,
    }

    print("\nDataset lengths:")
    for split_name, dataset in datasets.items():
        print(f"- {split_name}: {len(dataset)}")

    # -----------------------------------------------------
    # Class Distribution Checks
    # -----------------------------------------------------

    print("\nClass distributions:")
    class_presence_checks = {}

    for split_name, dataset in datasets.items():
        distribution = dataset.get_class_distribution()
        present = all_classes_present(distribution)
        class_presence_checks[split_name] = present

        print(f"\n{split_name}:")
        for class_name, count in distribution.items():
            print(f"  {class_name}: {count}")

        print("  all classes present:", present)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    loaders = {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }

    # -----------------------------------------------------
    # One-Batch Checks
    # -----------------------------------------------------

    print("\nBatch checks:")
    batch_checks = {}

    for split_name, loader in loaders.items():
        batch = next(iter(loader))
        batch_info = check_batch(batch, split_name)
        batch_checks[split_name] = batch_info

        print(f"\n{split_name} batch:")
        print("  image tensor shape:", batch_info["image_shape"])
        print("  label tensor shape:", batch_info["label_shape"])
        print("  image dtype:", batch_info["image_dtype"])
        print("  label dtype:", batch_info["label_dtype"])
        print("  image min:", batch_info["image_min"])
        print("  image max:", batch_info["image_max"])
        print("  labels in batch:", batch_info["labels"])
        print("  labels valid 0 to 3:", batch_info["labels_range_ok"])
        print("  batch check passed:", batch_info["batch_check_passed"])

    # Save visual transformed sample grid from train batch.
    train_images, train_labels = next(iter(train_loader))

    save_transformed_batch_grid(
        train_images,
        train_labels,
        TRANSFORMED_GRID_PATH,
        max_images=16,
    )

    print("\nTransformed train batch grid saved:")
    print(TRANSFORMED_GRID_PATH)

    # -----------------------------------------------------
    # Save Class Mapping JSON
    # -----------------------------------------------------

    class_mapping = {
        "target_classes": TARGET_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {
            str(idx): class_name
            for idx, class_name in IDX_TO_CLASS.items()
        },
    }

    save_json(CLASS_MAPPING_JSON_PATH, class_mapping)

    class_mapping_saved = CLASS_MAPPING_JSON_PATH.exists()

    print("\nClass mapping JSON saved:")
    print(CLASS_MAPPING_JSON_PATH)

    # -----------------------------------------------------
    # Final Pass/Fail Logic
    # -----------------------------------------------------

    all_datasets_loaded = all(
        len(dataset) > 0
        for dataset in datasets.values()
    )

    all_classes_in_all_splits = all(
        class_presence_checks.values()
    )

    all_batches_loaded = all(
        info["batch_check_passed"]
        for info in batch_checks.values()
    )

    expected_shape_ok = all(
        info["image_shape"] == [BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE]
        for info in batch_checks.values()
    )

    label_indices_valid = all(
        all(0 <= label < len(TARGET_CLASSES) for label in info["labels"])
        for info in batch_checks.values()
    )

    dataset_pipeline_check_passed = bool(
        all_datasets_loaded
        and all_classes_in_all_splits
        and all_batches_loaded
        and expected_shape_ok
        and label_indices_valid
        and class_mapping_saved
    )

    if dataset_pipeline_check_passed:
        recommended_next_step = (
            "Proceed to Prompt 7 - Baseline Transfer Learning Model Definition and Training Loop."
        )
    else:
        recommended_next_step = (
            "Do not proceed to training yet. Fix dataset pipeline issues first."
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
        "final_split_csv": str(FINAL_SPLIT_CSV),
        "target_classes": TARGET_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {
            str(idx): class_name
            for idx, class_name in IDX_TO_CLASS.items()
        },
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "num_workers": NUM_WORKERS,
        "dataset_summaries": dataset_summaries,
        "class_presence_checks": {
            split_name: bool(value)
            for split_name, value in class_presence_checks.items()
        },
        "batch_checks": batch_checks,
        "all_datasets_loaded": bool(all_datasets_loaded),
        "all_classes_in_all_splits": bool(all_classes_in_all_splits),
        "all_batches_loaded": bool(all_batches_loaded),
        "expected_shape_ok": bool(expected_shape_ok),
        "label_indices_valid": bool(label_indices_valid),
        "class_mapping_saved": bool(class_mapping_saved),
        "dataset_pipeline_check_passed": bool(dataset_pipeline_check_passed),
        "safe_to_proceed_to_prompt_7": bool(dataset_pipeline_check_passed),
        "output_files": {
            "class_mapping_json": str(CLASS_MAPPING_JSON_PATH),
            "dataset_pipeline_summary_json": str(PIPELINE_SUMMARY_JSON_PATH),
            "dataset_pipeline_report_md": str(PIPELINE_REPORT_PATH),
            "transformed_train_batch_grid": str(TRANSFORMED_GRID_PATH),
        },
        "recommended_next_step": recommended_next_step,
    }

    save_json(PIPELINE_SUMMARY_JSON_PATH, summary)

    # -----------------------------------------------------
    # Save Markdown Report
    # -----------------------------------------------------

    report_lines = []

    report_lines.append("# Dataset Pipeline Report")
    report_lines.append("")
    report_lines.append("## 1. Purpose")
    report_lines.append("")
    report_lines.append(
        "This report verifies the PyTorch dataset class, runtime transforms, "
        "DataLoaders, class mapping, and one-batch sanity checks. "
        "No model training is performed in this step."
    )

    report_lines.append("")
    report_lines.append("## 2. Project Information")
    report_lines.append("")
    report_lines.append(f"- Project root: `{PROJECT_ROOT}`")
    report_lines.append(f"- Device: `{DEVICE}`")
    report_lines.append(f"- Final split CSV: `{FINAL_SPLIT_CSV}`")
    report_lines.append(f"- Image size: `{IMAGE_SIZE}`")
    report_lines.append(f"- Batch size: `{BATCH_SIZE}`")
    report_lines.append(f"- Number of workers: `{NUM_WORKERS}`")

    report_lines.append("")
    report_lines.append("## 3. Class Mapping")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(class_mapping, indent=4))
    report_lines.append("```")

    report_lines.append("")
    report_lines.append("## 4. Dataset Lengths")
    report_lines.append("")
    report_lines.append("| Split | Length |")
    report_lines.append("|---|---:|")

    for split_name, dataset in datasets.items():
        report_lines.append(f"| {split_name} | {len(dataset)} |")

    report_lines.append("")
    report_lines.append("## 5. Class Distributions")
    report_lines.append("")

    for split_name, dataset in datasets.items():
        report_lines.append(markdown_table_from_dict(split_name, dataset.get_class_distribution()))
        report_lines.append("")

    report_lines.append("")
    report_lines.append("## 6. Class Presence Checks")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(class_presence_checks, indent=4))
    report_lines.append("```")

    report_lines.append("")
    report_lines.append("## 7. Batch Checks")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(batch_checks, indent=4))
    report_lines.append("```")

    report_lines.append("")
    report_lines.append("## 8. Output Files")
    report_lines.append("")
    report_lines.append(f"- Class mapping JSON: `{CLASS_MAPPING_JSON_PATH}`")
    report_lines.append(f"- Dataset pipeline summary JSON: `{PIPELINE_SUMMARY_JSON_PATH}`")
    report_lines.append(f"- Dataset pipeline report: `{PIPELINE_REPORT_PATH}`")
    report_lines.append(f"- Transformed train batch grid: `{TRANSFORMED_GRID_PATH}`")

    report_lines.append("")
    report_lines.append("## 9. Final Decision")
    report_lines.append("")
    report_lines.append(f"- Dataset pipeline check passed: `{dataset_pipeline_check_passed}`")
    report_lines.append(f"- Safe to proceed to Prompt 7: `{dataset_pipeline_check_passed}`")
    report_lines.append("")
    report_lines.append(recommended_next_step)

    PIPELINE_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    # -----------------------------------------------------
    # Console Summary
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("DATASET PIPELINE CHECK SUMMARY")
    print("=" * 80)

    print("\n1. Dataset loading")
    print("All datasets loaded:", all_datasets_loaded)

    print("\n2. Class presence")
    for split_name, present in class_presence_checks.items():
        print(f"{split_name}: all classes present -> {present}")

    print("\n3. Batch loading")
    for split_name, info in batch_checks.items():
        print(f"{split_name}: batch check passed -> {info['batch_check_passed']}")

    print("\n4. Shape and label checks")
    print("Expected shape OK:", expected_shape_ok)
    print("Label indices valid:", label_indices_valid)
    print("Class mapping saved:", class_mapping_saved)

    print("\n5. Output files")
    print("-", CLASS_MAPPING_JSON_PATH)
    print("-", PIPELINE_SUMMARY_JSON_PATH)
    print("-", PIPELINE_REPORT_PATH)
    print("-", TRANSFORMED_GRID_PATH)

    print("\n6. Final decision")
    print("Dataset pipeline check passed:", dataset_pipeline_check_passed)
    print("Safe to proceed to step 7:", dataset_pipeline_check_passed)
    print(recommended_next_step)

    print("\nDATASET PIPELINE CHECK COMPLETED.")


if __name__ == "__main__":
    main()
