from pathlib import Path
import json
import random

import pandas as pd
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
    MODELS_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    DEVICE,
)

from transforms import get_eval_transforms
from multilabel_dataset import MarineDebrisMultiLabelDataset
from multilabel_model import create_resnet18_multilabel_model
from multilabel_train_utils import (
    load_checkpoint,
    compute_multilabel_metrics_from_predictions,
)


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "multilabel_resnet18_layer4_threshold_tuning"

MULTILABEL_TRAIN_VAL_CSV = PROCESSED_RECORDS_DIR / "multilabel_train_val_records.csv"

LAYER4_CHECKPOINT_PATH = MODELS_DIR / "multilabel_resnet18_layer4_best.pth"

TARGET_CLASSES = ["plastic", "foam", "metal", "other_debris"]

TARGET_COLUMNS = [
    "multilabel_plastic",
    "multilabel_foam",
    "multilabel_metal",
    "multilabel_other_debris",
]

BASELINE_THRESHOLD = 0.5

THRESHOLD_GRID = [
    round(x, 2)
    for x in [
        0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.55,
        0.60, 0.65, 0.70, 0.75, 0.80,
        0.85, 0.90,
    ]
]


# =========================================================
# Output Files
# =========================================================

THRESHOLDS_JSON = REPORTS_DIR / "multilabel_layer4_tuned_thresholds.json"
SUMMARY_JSON = REPORTS_DIR / "multilabel_threshold_tuning_summary.json"
REPORT_MD = REPORTS_DIR / "multilabel_threshold_tuning_report.md"

THRESHOLD_SEARCH_CSV = REPORTS_DIR / "multilabel_threshold_search_results.csv"
BASELINE_VS_TUNED_CSV = REPORTS_DIR / "multilabel_threshold_baseline_vs_tuned.csv"
PER_CLASS_TUNED_METRICS_CSV = REPORTS_DIR / "multilabel_threshold_tuned_per_class_metrics.csv"

THRESHOLD_PLOT = PLOTS_DIR / "multilabel_threshold_tuning_per_class_f1.png"


# =========================================================
# Helper Functions
# =========================================================

def rel(path) -> str:
    """Return path relative to project root if possible."""
    path = Path(path)

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def set_seed(seed: int):
    """Set random seeds."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_validation_outputs(model, dataloader, device):
    """
    Collect validation targets, logits, and probabilities.

    Test split is not used.
    """
    model.eval()

    all_targets = []
    all_logits = []
    all_probabilities = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        probabilities = torch.sigmoid(logits)

        all_targets.append(targets.detach().cpu())
        all_logits.append(logits.detach().cpu())
        all_probabilities.append(probabilities.detach().cpu())

    targets_tensor = torch.cat(all_targets, dim=0)
    logits_tensor = torch.cat(all_logits, dim=0)
    probabilities_tensor = torch.cat(all_probabilities, dim=0)

    return targets_tensor, logits_tensor, probabilities_tensor


def apply_thresholds(probabilities: torch.Tensor, thresholds: list[float]) -> torch.Tensor:
    """
    Apply per-class thresholds to probabilities.

    probabilities shape:
        [num_samples, num_classes]

    thresholds:
        list of length num_classes
    """
    threshold_tensor = torch.tensor(thresholds, dtype=torch.float32).view(1, -1)

    predictions = (probabilities >= threshold_tensor).float()

    return predictions


def tune_threshold_for_one_class(
    y_true: torch.Tensor,
    probabilities: torch.Tensor,
    class_index: int,
    class_name: str,
):
    """
    Tune one class threshold by maximizing class F1.

    This is validation-only threshold tuning.
    """
    rows = []

    true_col = y_true[:, class_index].float()
    prob_col = probabilities[:, class_index].float()

    best_row = None

    for threshold in THRESHOLD_GRID:
        pred_col = (prob_col >= threshold).float()

        true_positive = ((true_col == 1) & (pred_col == 1)).sum().item()
        false_positive = ((true_col == 0) & (pred_col == 1)).sum().item()
        false_negative = ((true_col == 1) & (pred_col == 0)).sum().item()
        true_negative = ((true_col == 0) & (pred_col == 0)).sum().item()

        support = int((true_col == 1).sum().item())

        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        row = {
            "class_name": class_name,
            "class_index": int(class_index),
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
            "true_positive": int(true_positive),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_negative": int(true_negative),
        }

        rows.append(row)

        if best_row is None:
            best_row = row
        else:
            better_f1 = row["f1"] > best_row["f1"]
            same_f1_better_balance = (
                row["f1"] == best_row["f1"]
                and abs(row["precision"] - row["recall"])
                < abs(best_row["precision"] - best_row["recall"])
            )

            if better_f1 or same_f1_better_balance:
                best_row = row

    return best_row, rows


def tune_per_class_thresholds(y_true: torch.Tensor, probabilities: torch.Tensor):
    """
    Tune threshold independently for each class using validation set.
    """
    tuned_thresholds = []
    best_rows = []
    all_rows = []

    for class_index, class_name in enumerate(TARGET_CLASSES):
        best_row, rows = tune_threshold_for_one_class(
            y_true=y_true,
            probabilities=probabilities,
            class_index=class_index,
            class_name=class_name,
        )

        tuned_thresholds.append(float(best_row["threshold"]))
        best_rows.append(best_row)
        all_rows.extend(rows)

    return tuned_thresholds, best_rows, all_rows


def metrics_to_flat_row(model_label: str, thresholds: list[float], metrics: dict):
    """
    Convert metrics dictionary to one flat comparison row.
    """
    return {
        "model_label": model_label,
        "thresholds": json.dumps(thresholds),
        "subset_accuracy": metrics["subset_accuracy"],
        "hamming_accuracy": metrics["hamming_accuracy"],
        "micro_precision": metrics["micro_precision"],
        "micro_recall": metrics["micro_recall"],
        "micro_f1": metrics["micro_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
    }


def per_class_metrics_to_dataframe(per_class_metrics: dict) -> pd.DataFrame:
    """Convert per-class metrics dictionary to DataFrame."""
    rows = []

    for class_name in TARGET_CLASSES:
        item = per_class_metrics[class_name]

        rows.append({
            "class_name": class_name,
            "precision": item["precision"],
            "recall": item["recall"],
            "f1": item["f1"],
            "support": item["support"],
            "true_positive": item["true_positive"],
            "false_positive": item["false_positive"],
            "false_negative": item["false_negative"],
            "true_negative": item["true_negative"],
        })

    return pd.DataFrame(rows)


def plot_threshold_search_results(search_df: pd.DataFrame, output_path: Path):
    """
    Plot F1 vs threshold for each class.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for class_name in TARGET_CLASSES:
        class_df = search_df[search_df["class_name"] == class_name].copy()

        ax.plot(
            class_df["threshold"],
            class_df["f1"],
            marker="o",
            label=class_name,
        )

    ax.set_title("Validation F1 vs Threshold by Class")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert dataframe to Markdown table."""
    if df.empty:
        return "_No records found._"

    columns = list(df.columns)

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in df.iterrows():
        values = []

        for col in columns:
            text = str(row[col]).replace("\n", " ").replace("|", "/")
            values.append(text)

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def save_report(summary: dict, comparison_df: pd.DataFrame, best_thresholds_df: pd.DataFrame):
    """Save Markdown report."""
    lines = []

    lines.append("# Multi-Label Threshold Tuning Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report documents validation-only threshold tuning for the "
        "multi-label ResNet18 layer4 model."
    )

    lines.append("")
    lines.append("## 2. Important Notes")
    lines.append("")
    lines.append("- The test split is not loaded.")
    lines.append("- No model training is performed.")
    lines.append("- The official single-label model is untouched.")
    lines.append("- Thresholds are tuned using validation labels only.")
    lines.append("- These thresholds are experimental until evaluated on a final unseen test set.")

    lines.append("")
    lines.append("## 3. Model")
    lines.append("")
    lines.append(f"- Checkpoint: `{LAYER4_CHECKPOINT_PATH}`")
    lines.append(f"- Target classes: `{TARGET_CLASSES}`")
    lines.append(f"- Baseline threshold: `{BASELINE_THRESHOLD}`")
    lines.append(f"- Tuned thresholds: `{summary['tuned_thresholds_by_class']}`")

    lines.append("")
    lines.append("## 4. Best Thresholds Per Class")
    lines.append("")
    lines.append(dataframe_to_markdown(best_thresholds_df))

    lines.append("")
    lines.append("## 5. Baseline vs Tuned Thresholds")
    lines.append("")
    lines.append(dataframe_to_markdown(comparison_df))

    lines.append("")
    lines.append("## 6. Decision")
    lines.append("")
    lines.append(f"- Recommendation status: `{summary['recommendation_status']}`")
    lines.append("")
    lines.append(summary["recommended_next_step"])

    lines.append("")
    lines.append("## 7. Output Files")
    lines.append("")
    for name, path in summary["output_files"].items():
        lines.append(f"- {name}: `{path}`")

    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# Main
# =========================================================

def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("MULTI-LABEL VALIDATION THRESHOLD TUNING STARTED")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Device:", DEVICE)
    print("Validation records CSV:", MULTILABEL_TRAIN_VAL_CSV)
    print("Layer4 checkpoint:", LAYER4_CHECKPOINT_PATH)
    print("Target classes:", TARGET_CLASSES)
    print("Important: test split is not loaded.")
    print("Important: no training is performed.")

    if not MULTILABEL_TRAIN_VAL_CSV.exists():
        raise FileNotFoundError(
            f"Multi-label train/validation CSV not found: {MULTILABEL_TRAIN_VAL_CSV}"
        )

    if not LAYER4_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Multi-label layer4 checkpoint not found: {LAYER4_CHECKPOINT_PATH}"
        )

    # -----------------------------------------------------
    # Validation Dataset Only
    # -----------------------------------------------------

    validation_dataset = MarineDebrisMultiLabelDataset(
        records_csv=MULTILABEL_TRAIN_VAL_CSV,
        split="validation",
        transform=get_eval_transforms(),
        target_classes=TARGET_CLASSES,
        target_columns=TARGET_COLUMNS,
        require_readable=True,
        return_metadata=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("\nValidation dataset length:", len(validation_dataset))
    print("Validation positive counts:", validation_dataset.get_positive_counts())
    print("Validation label cardinality:", validation_dataset.get_label_cardinality())

    # -----------------------------------------------------
    # Load Model
    # -----------------------------------------------------

    model = create_resnet18_multilabel_model(
        num_classes=len(TARGET_CLASSES),
        freeze_backbone=True,
    )

    checkpoint = load_checkpoint(
        path=LAYER4_CHECKPOINT_PATH,
        device=DEVICE,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain model_state_dict.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # -----------------------------------------------------
    # Collect Validation Probabilities
    # -----------------------------------------------------

    y_true, logits, probabilities = collect_validation_outputs(
        model=model,
        dataloader=validation_loader,
        device=DEVICE,
    )

    # -----------------------------------------------------
    # Baseline Metrics at 0.5
    # -----------------------------------------------------

    baseline_thresholds = [BASELINE_THRESHOLD] * len(TARGET_CLASSES)
    baseline_predictions = apply_thresholds(probabilities, baseline_thresholds)

    baseline_metrics = compute_multilabel_metrics_from_predictions(
        y_true=y_true,
        y_pred=baseline_predictions,
        target_classes=TARGET_CLASSES,
    )

    # -----------------------------------------------------
    # Tune Per-Class Thresholds
    # -----------------------------------------------------

    tuned_thresholds, best_threshold_rows, search_rows = tune_per_class_thresholds(
        y_true=y_true,
        probabilities=probabilities,
    )

    tuned_predictions = apply_thresholds(probabilities, tuned_thresholds)

    tuned_metrics = compute_multilabel_metrics_from_predictions(
        y_true=y_true,
        y_pred=tuned_predictions,
        target_classes=TARGET_CLASSES,
    )

    # -----------------------------------------------------
    # Save CSV Outputs
    # -----------------------------------------------------

    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(THRESHOLD_SEARCH_CSV, index=False)

    best_thresholds_df = pd.DataFrame(best_threshold_rows)
    best_thresholds_df.to_csv(PER_CLASS_TUNED_METRICS_CSV, index=False)

    comparison_rows = [
        metrics_to_flat_row(
            model_label="layer4_threshold_0_5",
            thresholds=baseline_thresholds,
            metrics=baseline_metrics,
        ),
        metrics_to_flat_row(
            model_label="layer4_tuned_per_class_thresholds",
            thresholds=tuned_thresholds,
            metrics=tuned_metrics,
        ),
    ]

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(BASELINE_VS_TUNED_CSV, index=False)

    plot_threshold_search_results(
        search_df=search_df,
        output_path=THRESHOLD_PLOT,
    )

    # -----------------------------------------------------
    # Decision Logic
    # -----------------------------------------------------

    macro_f1_improved = tuned_metrics["macro_f1"] > baseline_metrics["macro_f1"]
    micro_f1_improved = tuned_metrics["micro_f1"] > baseline_metrics["micro_f1"]
    hamming_accuracy_improved = tuned_metrics["hamming_accuracy"] > baseline_metrics["hamming_accuracy"]
    subset_accuracy_improved = tuned_metrics["subset_accuracy"] > baseline_metrics["subset_accuracy"]

    if macro_f1_improved and micro_f1_improved:
        recommendation_status = "tuned_thresholds_clear_candidate"
        recommended_next_step = (
            "Use the tuned per-class thresholds with multilabel_resnet18_layer4_best.pth "
            "as the current multi-label candidate for dual-model inference. "
            "Do not tune using the test set."
        )

    elif macro_f1_improved:
        recommendation_status = "tuned_thresholds_macro_f1_only_improved"
        recommended_next_step = (
            "Tuned thresholds improved macro-F1 but not micro-F1. They may help class-balanced "
            "multi-label performance, but review the precision/recall tradeoff before final use."
        )

    else:
        recommendation_status = "keep_threshold_0_5"
        recommended_next_step = (
            "Tuned thresholds did not clearly improve validation macro-F1. Keep the fixed 0.5 "
            "threshold for the current multi-label model."
        )

    tuned_thresholds_by_class = {
        class_name: float(threshold)
        for class_name, threshold in zip(TARGET_CLASSES, tuned_thresholds)
    }

    # -----------------------------------------------------
    # Save Thresholds JSON
    # -----------------------------------------------------

    thresholds_config = {
        "model_checkpoint": rel(LAYER4_CHECKPOINT_PATH),
        "target_classes": TARGET_CLASSES,
        "target_columns": TARGET_COLUMNS,
        "threshold_type": "per_class_validation_tuned",
        "thresholds": tuned_thresholds_by_class,
        "threshold_vector": tuned_thresholds,
        "baseline_threshold": BASELINE_THRESHOLD,
        "validation_only": True,
        "test_set_used": False,
    }

    with open(THRESHOLDS_JSON, "w", encoding="utf-8") as f:
        json.dump(thresholds_config, f, indent=4)

    # -----------------------------------------------------
    # Save Summary JSON
    # -----------------------------------------------------

    summary = {
        "project_root": ".",
        "model_name": MODEL_NAME,
        "model_checkpoint": rel(LAYER4_CHECKPOINT_PATH),
        "validation_records_csv": rel(MULTILABEL_TRAIN_VAL_CSV),
        "target_classes": TARGET_CLASSES,
        "target_columns": TARGET_COLUMNS,
        "threshold_grid": THRESHOLD_GRID,
        "baseline_thresholds": baseline_thresholds,
        "tuned_thresholds": tuned_thresholds,
        "tuned_thresholds_by_class": tuned_thresholds_by_class,
        "validation_size": int(len(validation_dataset)),
        "validation_positive_counts": validation_dataset.get_positive_counts(),
        "validation_label_cardinality": float(validation_dataset.get_label_cardinality()),
        "baseline_metrics": baseline_metrics,
        "tuned_metrics": tuned_metrics,
        "macro_f1_improved": bool(macro_f1_improved),
        "micro_f1_improved": bool(micro_f1_improved),
        "hamming_accuracy_improved": bool(hamming_accuracy_improved),
        "subset_accuracy_improved": bool(subset_accuracy_improved),
        "recommendation_status": recommendation_status,
        "recommended_next_step": recommended_next_step,
        "training_performed": False,
        "test_set_used": False,
        "official_single_label_model_untouched": True,
        "output_files": {
            "thresholds_json": rel(THRESHOLDS_JSON),
            "summary_json": rel(SUMMARY_JSON),
            "report_md": rel(REPORT_MD),
            "threshold_search_csv": rel(THRESHOLD_SEARCH_CSV),
            "baseline_vs_tuned_csv": rel(BASELINE_VS_TUNED_CSV),
            "per_class_tuned_metrics_csv": rel(PER_CLASS_TUNED_METRICS_CSV),
            "threshold_plot": rel(THRESHOLD_PLOT),
        },
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    save_report(
        summary=summary,
        comparison_df=comparison_df,
        best_thresholds_df=best_thresholds_df,
    )

    # -----------------------------------------------------
    # Console Output
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("MULTI-LABEL THRESHOLD TUNING SUMMARY")
    print("=" * 80)

    print("\nBaseline thresholds:")
    print(baseline_thresholds)

    print("\nTuned thresholds:")
    print(tuned_thresholds_by_class)

    print("\nBaseline metrics:")
    print("Subset accuracy:", baseline_metrics["subset_accuracy"])
    print("Hamming accuracy:", baseline_metrics["hamming_accuracy"])
    print("Micro-F1:", baseline_metrics["micro_f1"])
    print("Macro-F1:", baseline_metrics["macro_f1"])
    print("Weighted-F1:", baseline_metrics["weighted_f1"])

    print("\nTuned metrics:")
    print("Subset accuracy:", tuned_metrics["subset_accuracy"])
    print("Hamming accuracy:", tuned_metrics["hamming_accuracy"])
    print("Micro-F1:", tuned_metrics["micro_f1"])
    print("Macro-F1:", tuned_metrics["macro_f1"])
    print("Weighted-F1:", tuned_metrics["weighted_f1"])

    print("\nImprovement flags:")
    print("Macro-F1 improved:", macro_f1_improved)
    print("Micro-F1 improved:", micro_f1_improved)
    print("Hamming accuracy improved:", hamming_accuracy_improved)
    print("Subset accuracy improved:", subset_accuracy_improved)

    print("\nRecommendation:")
    print(recommendation_status)
    print(recommended_next_step)

    print("\nSaved output files:")
    print("-", THRESHOLDS_JSON)
    print("-", SUMMARY_JSON)
    print("-", REPORT_MD)
    print("-", THRESHOLD_SEARCH_CSV)
    print("-", BASELINE_VS_TUNED_CSV)
    print("-", PER_CLASS_TUNED_METRICS_CSV)
    print("-", THRESHOLD_PLOT)

    print("\nMULTI-LABEL VALIDATION THRESHOLD TUNING COMPLETED.")


if __name__ == "__main__":
    main()
