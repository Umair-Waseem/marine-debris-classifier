from pathlib import Path
import json
import math

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    PROJECT_ROOT,
    FINAL_SPLIT_CSV,
    REPORTS_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    TARGET_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
)

from transforms import get_eval_transforms
from dataset import MarineDebrisDataset
from model import create_resnet18_model


# =========================================================
# Final Test Evaluation Configuration
# =========================================================

CHECKPOINT_PATH = MODELS_DIR / "resnet18_finetuned_layer4_best.pth"

MODEL_NAME = "resnet18_finetuned_layer4"

PREDICTIONS_CSV_PATH = REPORTS_DIR / "final_test_predictions.csv"
MISCLASSIFIED_CSV_PATH = REPORTS_DIR / "final_test_misclassified_samples.csv"
CORRECT_CSV_PATH = REPORTS_DIR / "final_test_correct_samples.csv"

CLASSIFICATION_REPORT_JSON_PATH = REPORTS_DIR / "final_test_classification_report.json"
EVALUATION_SUMMARY_JSON_PATH = REPORTS_DIR / "final_test_evaluation_summary.json"
EVALUATION_REPORT_MD_PATH = REPORTS_DIR / "final_test_evaluation_report.md"
CONFUSION_PAIRS_CSV_PATH = REPORTS_DIR / "final_test_confusion_pairs.csv"

CONFUSION_MATRIX_PLOT_PATH = PLOTS_DIR / "final_test_confusion_matrix.png"
NORMALIZED_CONFUSION_MATRIX_PLOT_PATH = PLOTS_DIR / "final_test_confusion_matrix_normalized.png"
MISCLASSIFIED_GRID_PATH = PLOTS_DIR / "final_test_misclassified_samples_grid.png"
CORRECT_GRID_PATH = PLOTS_DIR / "final_test_correct_samples_grid.png"

MAX_GRID_IMAGES = 16


# =========================================================
# Helper Functions
# =========================================================

def rel(path: Path) -> str:
    """Return path relative to project root if possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_checkpoint(path: Path, device):
    """
    Load PyTorch checkpoint safely.

    weights_only=False is used when supported because the checkpoint contains
    model weights plus metadata.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def safe_float(value):
    """Convert value to normal float."""
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_bool(value):
    """Convert common values to bool."""
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    text = str(value).strip().lower()

    return text in {"true", "1", "yes", "y"}


def extract_metadata_list(metadata_batch, batch_size: int):
    """
    Convert DataLoader-collated metadata dictionary into list of dictionaries.

    The dataset returns:
        image, label, metadata_dict

    PyTorch DataLoader collates metadata into a dictionary of lists/tensors.
    """
    metadata_list = []

    for index in range(batch_size):
        item = {}

        for key, value in metadata_batch.items():
            if isinstance(value, torch.Tensor):
                item[key] = value[index].item()
            elif isinstance(value, (list, tuple)):
                item[key] = value[index]
            else:
                item[key] = value

        metadata_list.append(item)

    return metadata_list


def compute_confusion_matrix(true_indices, predicted_indices, num_classes):
    """
    Create confusion matrix.

    Rows = true labels.
    Columns = predicted labels.
    """
    matrix = [
        [0 for _ in range(num_classes)]
        for _ in range(num_classes)
    ]

    for true_idx, pred_idx in zip(true_indices, predicted_indices):
        true_idx = int(true_idx)
        pred_idx = int(pred_idx)

        if 0 <= true_idx < num_classes and 0 <= pred_idx < num_classes:
            matrix[true_idx][pred_idx] += 1

    return matrix


def compute_metrics_from_confusion_matrix(confusion_matrix):
    """
    Compute per-class and aggregate classification metrics.

    Returns:
    - per-class precision, recall, F1, support
    - macro precision, recall, F1
    - weighted precision, recall, F1
    """
    num_classes = len(confusion_matrix)

    per_class_metrics = {}

    total_support = 0

    precision_values = []
    recall_values = []
    f1_values = []

    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0
    weighted_f1_sum = 0.0

    for class_idx in range(num_classes):
        true_positive = confusion_matrix[class_idx][class_idx]

        false_positive = sum(
            confusion_matrix[row_idx][class_idx]
            for row_idx in range(num_classes)
            if row_idx != class_idx
        )

        false_negative = sum(
            confusion_matrix[class_idx][col_idx]
            for col_idx in range(num_classes)
            if col_idx != class_idx
        )

        support = sum(confusion_matrix[class_idx])
        total_support += support

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

        class_name = IDX_TO_CLASS[class_idx]

        per_class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)

        weighted_precision_sum += precision * support
        weighted_recall_sum += recall * support
        weighted_f1_sum += f1 * support

    macro_precision = sum(precision_values) / max(len(precision_values), 1)
    macro_recall = sum(recall_values) / max(len(recall_values), 1)
    macro_f1 = sum(f1_values) / max(len(f1_values), 1)

    weighted_precision = weighted_precision_sum / max(total_support, 1)
    weighted_recall = weighted_recall_sum / max(total_support, 1)
    weighted_f1 = weighted_f1_sum / max(total_support, 1)

    return {
        "per_class_metrics": per_class_metrics,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
    }


def normalize_confusion_matrix(confusion_matrix):
    """
    Normalize confusion matrix row-wise.

    Each row sums to 1 if the class has support.
    """
    normalized = []

    for row in confusion_matrix:
        row_sum = sum(row)

        if row_sum == 0:
            normalized.append([0.0 for _ in row])
        else:
            normalized.append([float(value / row_sum) for value in row])

    return normalized


def plot_confusion_matrix(matrix, class_names, output_path: Path, title: str, normalized=False):
    """
    Save confusion matrix plot.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    image = ax.imshow(matrix, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for row_idx in range(len(class_names)):
        for col_idx in range(len(class_names)):
            value = matrix[row_idx][col_idx]

            if normalized:
                text = f"{value:.2f}"
            else:
                text = str(int(value))

            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def create_prediction_grid(df: pd.DataFrame, output_path: Path, title: str, max_images: int = 16):
    """
    Save visual grid of predictions.

    This only creates a review plot. It does not save resized dataset images.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        fig = plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No samples available.", ha="center", va="center")
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    sample_df = df.head(max_images).copy()

    n_images = len(sample_df)
    n_cols = 4
    n_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    if hasattr(axes, "flatten"):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for ax in axes_flat:
        ax.axis("off")

    for ax, (_, row) in zip(axes_flat, sample_df.iterrows()):
        image_path = Path(str(row["image_path"]))

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image.thumbnail((256, 256))
                ax.imshow(image)
        except Exception:
            ax.text(0.5, 0.5, "Image not readable", ha="center", va="center")

        true_label = row["true_label"]
        predicted_label = row["predicted_label"]
        confidence = safe_float(row["confidence"])

        ax.set_title(
            f"T: {true_label}\nP: {predicted_label}\nConf: {confidence:.2f}",
            fontsize=9,
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 50):
    """Convert DataFrame to Markdown table."""
    if df is None or df.empty:
        return "_No records found._"

    small_df = df.head(max_rows).copy()
    columns = list(small_df.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []

    for _, row in small_df.iterrows():
        values = []

        for col in columns:
            text = str(row[col]).replace("\n", " ").replace("|", "/")
            values.append(text)

        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator] + rows)


def metric_rows_from_per_class_metrics(per_class_metrics):
    """Create rows for Markdown table."""
    rows = []

    for class_name in TARGET_CLASSES:
        metrics = per_class_metrics[class_name]

        rows.append({
            "class": class_name,
            "precision": round(metrics["precision"], 6),
            "recall": round(metrics["recall"], 6),
            "f1": round(metrics["f1"], 6),
            "support": metrics["support"],
        })

    return pd.DataFrame(rows)


def build_confusion_pair_dataframe(predictions_df: pd.DataFrame):
    """
    Build confusion-pair counts for incorrect predictions.
    """
    incorrect_df = predictions_df[predictions_df["correct"] == False].copy()

    if incorrect_df.empty:
        return pd.DataFrame(
            columns=["true_label", "predicted_label", "count"]
        )

    pair_df = (
        incorrect_df
        .groupby(["true_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    return pair_df


def build_error_summary_by_column(predictions_df: pd.DataFrame, column_name: str):
    """
    Summarize correct/incorrect counts by a column.
    """
    if predictions_df.empty or column_name not in predictions_df.columns:
        return {}

    summary = {}

    grouped = predictions_df.groupby([column_name, "correct"]).size().reset_index(name="count")

    for _, row in grouped.iterrows():
        key = str(row[column_name])
        correct_key = "correct" if bool(row["correct"]) else "incorrect"

        if key not in summary:
            summary[key] = {"correct": 0, "incorrect": 0, "total": 0}

        summary[key][correct_key] += int(row["count"])
        summary[key]["total"] += int(row["count"])

    for key, item in summary.items():
        total = max(item["total"], 1)
        item["accuracy"] = float(item["correct"] / total)
        item["error_rate"] = float(item["incorrect"] / total)

    return summary


# =========================================================
# Main Final Test Evaluation
# =========================================================

def main():
    print("=" * 80)
    print("FINAL TEST EVALUATION STARTED")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Device:", DEVICE)
    print("Selected checkpoint:", CHECKPOINT_PATH)
    print("Final split CSV:", FINAL_SPLIT_CSV)

    if not FINAL_SPLIT_CSV.exists():
        raise FileNotFoundError(f"Final split CSV not found: {FINAL_SPLIT_CSV}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Selected checkpoint not found: {CHECKPOINT_PATH}")

    # -----------------------------------------------------
    # Load checkpoint and model
    # -----------------------------------------------------

    checkpoint = load_checkpoint(CHECKPOINT_PATH, DEVICE)

    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")

    model = create_resnet18_model(
        num_classes=len(TARGET_CLASSES),
        freeze_backbone=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    # -----------------------------------------------------
    # Load test dataset only
    # -----------------------------------------------------

    test_dataset = MarineDebrisDataset(
        records_csv=FINAL_SPLIT_CSV,
        split="test",
        transform=get_eval_transforms(),
        target_classes=TARGET_CLASSES,
        class_to_idx=CLASS_TO_IDX,
        require_readable=True,
        include_ambiguous=True,
        return_metadata=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_distribution = test_dataset.get_class_distribution()

    print("\nTest dataset size:", len(test_dataset))
    print("Test class distribution:")
    for class_name, count in test_distribution.items():
        print(f"- {class_name}: {count}")

    # -----------------------------------------------------
    # Final test inference
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    all_true_indices = []
    all_predicted_indices = []
    prediction_rows = []

    with torch.no_grad():
        for images, labels, metadata_batch in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probabilities = torch.softmax(outputs, dim=1)
            confidence_values, predicted_indices = torch.max(probabilities, dim=1)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            correct_tensor = predicted_indices == labels
            total_correct += correct_tensor.sum().item()

            metadata_list = extract_metadata_list(metadata_batch, batch_size)

            for index in range(batch_size):
                true_idx = int(labels[index].detach().cpu().item())
                pred_idx = int(predicted_indices[index].detach().cpu().item())
                confidence = float(confidence_values[index].detach().cpu().item())
                correct = bool(correct_tensor[index].detach().cpu().item())

                true_label = IDX_TO_CLASS[true_idx]
                predicted_label = IDX_TO_CLASS[pred_idx]

                metadata = metadata_list[index]

                all_true_indices.append(true_idx)
                all_predicted_indices.append(pred_idx)

                prediction_rows.append({
                    "filename": str(metadata.get("filename", "")),
                    "image_path": str(metadata.get("image_path", "")),
                    "true_label": true_label,
                    "true_index": true_idx,
                    "predicted_label": predicted_label,
                    "predicted_index": pred_idx,
                    "confidence": confidence,
                    "correct": correct,
                    "caution_flag": str(metadata.get("caution_flag", "")),
                    "is_ambiguous": safe_bool(metadata.get("is_ambiguous", False)),
                    "final_split": str(metadata.get("final_split", "test")),
                })

    test_loss = total_loss / max(total_samples, 1)
    test_accuracy = total_correct / max(total_samples, 1)
    total_incorrect = total_samples - total_correct

    predictions_df = pd.DataFrame(prediction_rows)
    predictions_df.to_csv(PREDICTIONS_CSV_PATH, index=False)

    misclassified_df = predictions_df[predictions_df["correct"] == False].copy()
    correct_df = predictions_df[predictions_df["correct"] == True].copy()

    misclassified_df.to_csv(MISCLASSIFIED_CSV_PATH, index=False)
    correct_df.to_csv(CORRECT_CSV_PATH, index=False)

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    confusion_matrix = compute_confusion_matrix(
        true_indices=all_true_indices,
        predicted_indices=all_predicted_indices,
        num_classes=len(TARGET_CLASSES),
    )

    normalized_confusion_matrix = normalize_confusion_matrix(confusion_matrix)

    metric_result = compute_metrics_from_confusion_matrix(confusion_matrix)

    per_class_metrics = metric_result["per_class_metrics"]
    macro_precision = metric_result["macro_precision"]
    macro_recall = metric_result["macro_recall"]
    macro_f1 = metric_result["macro_f1"]
    weighted_precision = metric_result["weighted_precision"]
    weighted_recall = metric_result["weighted_recall"]
    weighted_f1 = metric_result["weighted_f1"]

    classification_report = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "normalized_confusion_matrix": normalized_confusion_matrix,
        "target_classes": TARGET_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {
            str(index): class_name
            for index, class_name in IDX_TO_CLASS.items()
        },
    }

    with open(CLASSIFICATION_REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(classification_report, f, indent=4)

    # -----------------------------------------------------
    # Error analysis
    # -----------------------------------------------------

    confusion_pairs_df = build_confusion_pair_dataframe(predictions_df)
    confusion_pairs_df.to_csv(CONFUSION_PAIRS_CSV_PATH, index=False)

    caution_flag_error_summary = build_error_summary_by_column(
        predictions_df,
        "caution_flag",
    )

    ambiguity_error_summary = build_error_summary_by_column(
        predictions_df,
        "is_ambiguous",
    )

    class_error_counts = {}

    for class_name in TARGET_CLASSES:
        class_df = predictions_df[predictions_df["true_label"] == class_name]

        class_total = int(len(class_df))
        class_correct = int(class_df["correct"].sum()) if class_total > 0 else 0
        class_incorrect = int(class_total - class_correct)

        class_error_counts[class_name] = {
            "total": class_total,
            "correct": class_correct,
            "incorrect": class_incorrect,
            "accuracy": float(class_correct / class_total) if class_total > 0 else 0.0,
            "error_rate": float(class_incorrect / class_total) if class_total > 0 else 0.0,
        }

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------

    plot_confusion_matrix(
        matrix=confusion_matrix,
        class_names=TARGET_CLASSES,
        output_path=CONFUSION_MATRIX_PLOT_PATH,
        title="Final Test Confusion Matrix",
        normalized=False,
    )

    plot_confusion_matrix(
        matrix=normalized_confusion_matrix,
        class_names=TARGET_CLASSES,
        output_path=NORMALIZED_CONFUSION_MATRIX_PLOT_PATH,
        title="Final Test Normalized Confusion Matrix",
        normalized=True,
    )

    create_prediction_grid(
        df=misclassified_df,
        output_path=MISCLASSIFIED_GRID_PATH,
        title="Final Test Misclassified Samples",
        max_images=MAX_GRID_IMAGES,
    )

    create_prediction_grid(
        df=correct_df,
        output_path=CORRECT_GRID_PATH,
        title="Final Test Correct Samples",
        max_images=MAX_GRID_IMAGES,
    )

    # -----------------------------------------------------
    # Final recommendation
    # -----------------------------------------------------

    if total_samples > 0:
        recommended_next_step = (
            "Proceed to Prompt 10 - Inference Pipeline, Streamlit App, "
            "Final Documentation, and Submission Audit. Report final test results honestly "
            "with class-wise metrics, macro-F1, confusion matrix, and limitations."
        )
    else:
        recommended_next_step = (
            "Final test evaluation did not run correctly because the test set is empty. "
            "Fix the dataset split before proceeding."
        )

    # -----------------------------------------------------
    # Save JSON summary
    # -----------------------------------------------------

    summary = {
        "checkpoint_path": rel(CHECKPOINT_PATH),
        "model_name": MODEL_NAME,
        "test_set_used": True,
        "train_used": False,
        "validation_used_for_tuning": False,
        "test_size": int(total_samples),
        "test_class_distribution": test_distribution,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion_matrix,
        "normalized_confusion_matrix": normalized_confusion_matrix,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {
            str(index): class_name
            for index, class_name in IDX_TO_CLASS.items()
        },
        "target_classes": TARGET_CLASSES,
        "total_correct": int(total_correct),
        "total_incorrect": int(total_incorrect),
        "class_error_counts": class_error_counts,
        "caution_flag_error_summary": caution_flag_error_summary,
        "ambiguity_error_summary": ambiguity_error_summary,
        "most_common_confusion_pairs": confusion_pairs_df.to_dict(orient="records"),
        "important_limitations": [
            "Dataset is small.",
            "Metal class has only 3 test images.",
            "Many labels are derived from ambiguous multi-object images.",
            "Results should not be overclaimed as real-world deployment performance.",
            "Test results must not be used to tune the model further.",
        ],
        "output_files": {
            "predictions_csv": rel(PREDICTIONS_CSV_PATH),
            "misclassified_csv": rel(MISCLASSIFIED_CSV_PATH),
            "correct_csv": rel(CORRECT_CSV_PATH),
            "classification_report_json": rel(CLASSIFICATION_REPORT_JSON_PATH),
            "evaluation_summary_json": rel(EVALUATION_SUMMARY_JSON_PATH),
            "evaluation_report_md": rel(EVALUATION_REPORT_MD_PATH),
            "confusion_pairs_csv": rel(CONFUSION_PAIRS_CSV_PATH),
            "confusion_matrix_plot": rel(CONFUSION_MATRIX_PLOT_PATH),
            "normalized_confusion_matrix_plot": rel(NORMALIZED_CONFUSION_MATRIX_PLOT_PATH),
            "misclassified_grid": rel(MISCLASSIFIED_GRID_PATH),
            "correct_grid": rel(CORRECT_GRID_PATH),
        },
        "recommended_next_step": recommended_next_step,
    }

    with open(EVALUATION_SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # -----------------------------------------------------
    # Markdown report
    # -----------------------------------------------------

    per_class_metrics_df = metric_rows_from_per_class_metrics(per_class_metrics)

    markdown_lines = []

    markdown_lines.append("# Final Test Evaluation Report")
    markdown_lines.append("")
    markdown_lines.append("## 1. Purpose")
    markdown_lines.append("")
    markdown_lines.append(
        "This report presents the final test evaluation of the selected fine-tuned "
        "ResNet18 model. This is the first final test evaluation step. The model is "
        "not trained or tuned in this script."
    )

    markdown_lines.append("")
    markdown_lines.append("## 2. Selected Checkpoint")
    markdown_lines.append("")
    markdown_lines.append(f"`{rel(CHECKPOINT_PATH)}`")

    markdown_lines.append("")
    markdown_lines.append("## 3. Test Set Information")
    markdown_lines.append("")
    markdown_lines.append(f"- Test set size: `{total_samples}`")
    markdown_lines.append("- Test class distribution:")
    markdown_lines.append("")
    for class_name, count in test_distribution.items():
        markdown_lines.append(f"  - {class_name}: {count}")

    markdown_lines.append("")
    markdown_lines.append("## 4. Final Test Metrics")
    markdown_lines.append("")
    markdown_lines.append(f"- Test loss: `{test_loss:.6f}`")
    markdown_lines.append(f"- Test accuracy: `{test_accuracy:.6f}`")
    markdown_lines.append(f"- Macro precision: `{macro_precision:.6f}`")
    markdown_lines.append(f"- Macro recall: `{macro_recall:.6f}`")
    markdown_lines.append(f"- Macro F1: `{macro_f1:.6f}`")
    markdown_lines.append(f"- Weighted precision: `{weighted_precision:.6f}`")
    markdown_lines.append(f"- Weighted recall: `{weighted_recall:.6f}`")
    markdown_lines.append(f"- Weighted F1: `{weighted_f1:.6f}`")

    markdown_lines.append("")
    markdown_lines.append("## 5. Per-Class Precision / Recall / F1 / Support")
    markdown_lines.append("")
    markdown_lines.append(dataframe_to_markdown(per_class_metrics_df))

    markdown_lines.append("")
    markdown_lines.append("## 6. Confusion Matrix")
    markdown_lines.append("")
    markdown_lines.append("Rows are true labels and columns are predicted labels.")
    markdown_lines.append("")
    markdown_lines.append("```json")
    markdown_lines.append(json.dumps(confusion_matrix, indent=4))
    markdown_lines.append("```")

    markdown_lines.append("")
    markdown_lines.append("## 7. Most Common Confusion Pairs")
    markdown_lines.append("")
    markdown_lines.append(dataframe_to_markdown(confusion_pairs_df))

    markdown_lines.append("")
    markdown_lines.append("## 8. Misclassification Analysis")
    markdown_lines.append("")
    markdown_lines.append(f"- Correct predictions: `{total_correct}`")
    markdown_lines.append(f"- Incorrect predictions: `{total_incorrect}`")
    markdown_lines.append("")
    markdown_lines.append("### Class Error Counts")
    markdown_lines.append("")
    markdown_lines.append("```json")
    markdown_lines.append(json.dumps(class_error_counts, indent=4))
    markdown_lines.append("```")

    markdown_lines.append("")
    markdown_lines.append("## 9. Effect of Caution Flag on Errors")
    markdown_lines.append("")
    markdown_lines.append("```json")
    markdown_lines.append(json.dumps(caution_flag_error_summary, indent=4))
    markdown_lines.append("```")

    markdown_lines.append("")
    markdown_lines.append("## 10. Effect of Ambiguity on Errors")
    markdown_lines.append("")
    markdown_lines.append("```json")
    markdown_lines.append(json.dumps(ambiguity_error_summary, indent=4))
    markdown_lines.append("```")

    markdown_lines.append("")
    markdown_lines.append("## 11. Visual Outputs")
    markdown_lines.append("")
    markdown_lines.append(f"- Confusion matrix: `{rel(CONFUSION_MATRIX_PLOT_PATH)}`")
    markdown_lines.append(
        f"- Normalized confusion matrix: `{rel(NORMALIZED_CONFUSION_MATRIX_PLOT_PATH)}`"
    )
    markdown_lines.append(f"- Misclassified samples grid: `{rel(MISCLASSIFIED_GRID_PATH)}`")
    markdown_lines.append(f"- Correct samples grid: `{rel(CORRECT_GRID_PATH)}`")

    markdown_lines.append("")
    markdown_lines.append("## 12. Important Limitations")
    markdown_lines.append("")
    markdown_lines.append("- Dataset is small.")
    markdown_lines.append("- Metal class has only 3 test images.")
    markdown_lines.append("- Many labels come from dominant-label conversion of multi-object images.")
    markdown_lines.append("- Results should not be described as real-world deployment performance.")
    markdown_lines.append("- Test results must not be used for further model selection or tuning.")

    markdown_lines.append("")
    markdown_lines.append("## 13. Final Recommendation")
    markdown_lines.append("")
    markdown_lines.append(recommended_next_step)

    EVALUATION_REPORT_MD_PATH.write_text("\n".join(markdown_lines), encoding="utf-8")

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION SUMMARY")
    print("=" * 80)

    print("\n1. Selected checkpoint")
    print(CHECKPOINT_PATH)

    print("\n2. Test dataset")
    print("Test size:", total_samples)
    print("Test class distribution:")
    for class_name, count in test_distribution.items():
        print(f"- {class_name}: {count}")

    print("\n3. Final test metrics")
    print("Test loss:", test_loss)
    print("Test accuracy:", test_accuracy)
    print("Macro precision:", macro_precision)
    print("Macro recall:", macro_recall)
    print("Macro F1:", macro_f1)
    print("Weighted precision:", weighted_precision)
    print("Weighted recall:", weighted_recall)
    print("Weighted F1:", weighted_f1)

    print("\n4. Per-class metrics")
    for class_name, metrics in per_class_metrics.items():
        print(f"{class_name}: {metrics}")

    print("\n5. Confusion matrix")
    print("Rows = true labels, columns = predicted labels")
    for row in confusion_matrix:
        print(row)

    print("\n6. Prediction counts")
    print("Correct:", total_correct)
    print("Incorrect:", total_incorrect)

    print("\n7. Saved output files")
    print("-", PREDICTIONS_CSV_PATH)
    print("-", MISCLASSIFIED_CSV_PATH)
    print("-", CORRECT_CSV_PATH)
    print("-", CLASSIFICATION_REPORT_JSON_PATH)
    print("-", EVALUATION_SUMMARY_JSON_PATH)
    print("-", EVALUATION_REPORT_MD_PATH)
    print("-", CONFUSION_PAIRS_CSV_PATH)
    print("-", CONFUSION_MATRIX_PLOT_PATH)
    print("-", NORMALIZED_CONFUSION_MATRIX_PLOT_PATH)
    print("-", MISCLASSIFIED_GRID_PATH)
    print("-", CORRECT_GRID_PATH)

    print("\n8. Final recommendation")
    print(recommended_next_step)

    print("\nFINAL TEST EVALUATION COMPLETED.")


if __name__ == "__main__":
    main()
