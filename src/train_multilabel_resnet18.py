import json
import random
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    PROJECT_ROOT,
    PROCESSED_RECORDS_DIR,
    REPORTS_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEVICE,
)

from transforms import get_train_transforms, get_eval_transforms
from multilabel_dataset import MarineDebrisMultiLabelDataset
from multilabel_model import (
    create_resnet18_multilabel_model,
    count_model_parameters,
)
from multilabel_train_utils import (
    compute_pos_weight_from_dataset,
    train_multilabel_one_epoch,
    validate_multilabel_one_epoch,
    save_checkpoint,
    plot_multilabel_training_curves,
    save_history_csv,
)


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "multilabel_resnet18_fc"
TRAINING_STRATEGY = "resnet18_multilabel_fc_only_frozen_backbone"

MULTILABEL_TRAIN_VAL_CSV = PROCESSED_RECORDS_DIR / "multilabel_train_val_records.csv"

TARGET_CLASSES = ["plastic", "foam", "metal", "other_debris"]

TARGET_COLUMNS = [
    "multilabel_plastic",
    "multilabel_foam",
    "multilabel_metal",
    "multilabel_other_debris",
]

THRESHOLD = 0.5

EPOCHS = 15
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BEST_CHECKPOINT_PATH = MODELS_DIR / "multilabel_resnet18_fc_best.pth"
LAST_CHECKPOINT_PATH = MODELS_DIR / "multilabel_resnet18_fc_last.pth"

HISTORY_CSV_PATH = REPORTS_DIR / "multilabel_resnet18_fc_training_history.csv"
SUMMARY_JSON_PATH = REPORTS_DIR / "multilabel_resnet18_fc_training_summary.json"
REPORT_MD_PATH = REPORTS_DIR / "multilabel_resnet18_fc_training_report.md"
CURVES_PATH = PLOTS_DIR / "multilabel_resnet18_fc_training_curves.png"


# =========================================================
# Helper Functions
# =========================================================

def set_seed(seed):
    """Set reproducible seeds."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def markdown_table_from_dict(title, data):
    """Create simple Markdown table from dictionary."""
    lines = [
        f"### {title}",
        "",
        "| Item | Value |",
        "|---|---:|",
    ]

    for key, value in data.items():
        lines.append(f"| {key} | {value} |")

    return "\n".join(lines)


def dataframe_to_markdown(df: pd.DataFrame):
    """Convert DataFrame to Markdown table."""
    if df is None or df.empty:
        return "_No records found._"

    columns = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in df.iterrows():
        values = []

        for col in columns:
            values.append(str(row[col]).replace("|", "/"))

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def per_class_metrics_to_dataframe(per_class_metrics):
    """Convert per-class metrics dict to DataFrame."""
    rows = []

    for class_name in TARGET_CLASSES:
        metrics = per_class_metrics[class_name]

        rows.append({
            "class": class_name,
            "precision": round(metrics["precision"], 6),
            "recall": round(metrics["recall"], 6),
            "f1": round(metrics["f1"], 6),
            "support": metrics["support"],
            "tp": metrics["true_positive"],
            "fp": metrics["false_positive"],
            "fn": metrics["false_negative"],
            "tn": metrics["true_negative"],
        })

    return pd.DataFrame(rows)


def make_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_loss,
    current_val_loss,
    current_val_macro_f1,
    current_val_micro_f1,
    pos_weight,
    history,
    current_metrics,
    train_summary,
    validation_summary,
    training_completed,
    early_stopped,
):
    """Create checkpoint dictionary."""

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "current_val_loss": float(current_val_loss),
        "current_val_macro_f1": float(current_val_macro_f1),
        "current_val_micro_f1": float(current_val_micro_f1),
        "target_classes": list(TARGET_CLASSES),
        "target_columns": list(TARGET_COLUMNS),
        "threshold": float(THRESHOLD),
        "pos_weight": [
            float(x)
            for x in pos_weight.detach().cpu().tolist()
        ],
        "model_name": MODEL_NAME,
        "training_strategy": TRAINING_STRATEGY,
        "image_size": int(IMAGE_SIZE),
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": float(LEARNING_RATE),
        "weight_decay": float(WEIGHT_DECAY),
        "history": history,
        "current_metrics": current_metrics,
        "train_dataset_summary": train_summary,
        "validation_dataset_summary": validation_summary,
        "training_completed": bool(training_completed),
        "early_stopped": bool(early_stopped),
        "test_set_used": False,
    }

    return checkpoint


def save_markdown_report(summary):
    """Save Markdown report."""

    best_per_class_df = per_class_metrics_to_dataframe(
        summary["best_validation_per_class_metrics"]
    )

    lines = []

    lines.append("# Multi-Label ResNet18 FC Training Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report documents the first experimental multi-label ResNet18 model. "
        "The model predicts the presence or absence of four debris classes using "
        "multi-hot targets and BCEWithLogitsLoss."
    )

    lines.append("")
    lines.append("## 2. Important Notes")
    lines.append("")
    lines.append("- This is an experimental multi-label branch.")
    lines.append("- The official single-label model is not overwritten.")
    lines.append("- The test split is not loaded or evaluated.")
    lines.append("- Threshold is fixed at `0.5` in this first baseline.")
    lines.append("- No threshold tuning is performed.")

    lines.append("")
    lines.append("## 3. Dataset Summary")
    lines.append("")
    lines.append(markdown_table_from_dict("Train Dataset", summary["train_positive_counts"]))
    lines.append("")
    lines.append(markdown_table_from_dict("Validation Dataset", summary["validation_positive_counts"]))

    lines.append("")
    lines.append("## 4. Positive Weights")
    lines.append("")
    pos_weight_dict = {
        class_name: summary["pos_weight"][index]
        for index, class_name in enumerate(TARGET_CLASSES)
    }
    lines.append(markdown_table_from_dict("BCE pos_weight", pos_weight_dict))

    lines.append("")
    lines.append("## 5. Model and Training Settings")
    lines.append("")
    settings = {
        "Model name": MODEL_NAME,
        "Training strategy": TRAINING_STRATEGY,
        "Epochs requested": EPOCHS,
        "Epochs completed": summary["epochs_completed"],
        "Early stopped": summary["early_stopped"],
        "Learning rate": LEARNING_RATE,
        "Weight decay": WEIGHT_DECAY,
        "Batch size": BATCH_SIZE,
        "Threshold": THRESHOLD,
        "Device": summary["device"],
    }
    lines.append(markdown_table_from_dict("Settings", settings))

    lines.append("")
    lines.append("## 6. Best Validation Results")
    lines.append("")
    best_results = {
        "Best epoch": summary["best_epoch"],
        "Best validation loss": summary["best_val_loss"],
        "Best validation subset accuracy": summary["best_validation_subset_accuracy"],
        "Best validation hamming accuracy": summary["best_validation_hamming_accuracy"],
        "Best validation micro-F1": summary["best_validation_micro_f1"],
        "Best validation macro-F1": summary["best_validation_macro_f1"],
        "Best validation weighted-F1": summary["best_validation_weighted_f1"],
    }
    lines.append(markdown_table_from_dict("Best Validation Metrics", best_results))

    lines.append("")
    lines.append("## 7. Best Per-Class Validation Metrics")
    lines.append("")
    lines.append(dataframe_to_markdown(best_per_class_df))

    lines.append("")
    lines.append("## 8. Important Limitations")
    lines.append("")
    lines.append("- This model has not been tested on the held-out test split.")
    lines.append("- Threshold is fixed at 0.5 and may not be optimal for every class.")
    lines.append("- Multi-label targets are derived from mapped dataset annotations.")
    lines.append("- This branch does not replace the official single-label result yet.")

    lines.append("")
    lines.append("## 9. Output Files")
    lines.append("")
    for name, path in summary["output_files"].items():
        lines.append(f"- {name}: `{path}`")

    lines.append("")
    lines.append("## 10. Recommendation")
    lines.append("")
    lines.append(summary["recommended_next_step"])

    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# Main Training Script
# =========================================================

def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("MULTI-LABEL RESNET18 FC TRAINING STARTED")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Device:", DEVICE)
    print("Records CSV:", MULTILABEL_TRAIN_VAL_CSV)
    print("Target classes:", TARGET_CLASSES)
    print("Target columns:", TARGET_COLUMNS)
    print("Threshold:", THRESHOLD)
    print("Important: test split is not loaded.")

    if not MULTILABEL_TRAIN_VAL_CSV.exists():
        raise FileNotFoundError(
            f"Multi-label train/validation CSV not found: {MULTILABEL_TRAIN_VAL_CSV}"
        )

    # -----------------------------------------------------
    # Datasets and DataLoaders
    # -----------------------------------------------------

    train_dataset = MarineDebrisMultiLabelDataset(
        records_csv=MULTILABEL_TRAIN_VAL_CSV,
        split="train",
        transform=get_train_transforms(),
        target_classes=TARGET_CLASSES,
        target_columns=TARGET_COLUMNS,
        require_readable=True,
        return_metadata=False,
    )

    validation_dataset = MarineDebrisMultiLabelDataset(
        records_csv=MULTILABEL_TRAIN_VAL_CSV,
        split="validation",
        transform=get_eval_transforms(),
        target_classes=TARGET_CLASSES,
        target_columns=TARGET_COLUMNS,
        require_readable=True,
        return_metadata=False,
    )

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

    train_summary = train_dataset.get_summary()
    validation_summary = validation_dataset.get_summary()

    print("\nDataset lengths:")
    print("Train:", len(train_dataset))
    print("Validation:", len(validation_dataset))

    print("\nTrain positive counts:")
    print(train_dataset.get_positive_counts())

    print("\nValidation positive counts:")
    print(validation_dataset.get_positive_counts())

    print("\nTrain label cardinality:", train_dataset.get_label_cardinality())
    print("Validation label cardinality:", validation_dataset.get_label_cardinality())

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = create_resnet18_multilabel_model(
        num_classes=len(TARGET_CLASSES),
        freeze_backbone=True,
    )

    parameter_counts = count_model_parameters(model)

    print("\nModel parameter counts:")
    print(parameter_counts)

    model = model.to(DEVICE)

    # -----------------------------------------------------
    # Loss and Optimizer
    # -----------------------------------------------------

    pos_weight = compute_pos_weight_from_dataset(
        dataset=train_dataset,
        target_classes=TARGET_CLASSES,
    ).to(DEVICE)

    print("\nBCE pos_weight:")
    for class_name, weight in zip(TARGET_CLASSES, pos_weight.detach().cpu().tolist()):
        print(f"- {class_name}: {weight:.6f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Training Loop
    # -----------------------------------------------------

    history = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics = None

    epochs_without_improvement = 0
    early_stopped = False
    training_completed = False

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "-" * 80)
        print(f"Epoch {epoch}/{EPOCHS}")
        print("-" * 80)

        train_loss = train_multilabel_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
        )

        validation_loss, validation_metrics, _, _, _ = validate_multilabel_one_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=DEVICE,
            threshold=THRESHOLD,
            target_classes=TARGET_CLASSES,
            epoch=epoch,
        )

        improved = validation_loss < best_val_loss

        if improved:
            best_val_loss = validation_loss
            best_epoch = epoch
            best_metrics = validation_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history_row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
            "validation_subset_accuracy": float(validation_metrics["subset_accuracy"]),
            "validation_hamming_accuracy": float(validation_metrics["hamming_accuracy"]),
            "validation_micro_precision": float(validation_metrics["micro_precision"]),
            "validation_micro_recall": float(validation_metrics["micro_recall"]),
            "validation_micro_f1": float(validation_metrics["micro_f1"]),
            "validation_macro_precision": float(validation_metrics["macro_precision"]),
            "validation_macro_recall": float(validation_metrics["macro_recall"]),
            "validation_macro_f1": float(validation_metrics["macro_f1"]),
            "validation_weighted_precision": float(validation_metrics["weighted_precision"]),
            "validation_weighted_recall": float(validation_metrics["weighted_recall"]),
            "validation_weighted_f1": float(validation_metrics["weighted_f1"]),
            "best_val_loss_so_far": float(best_val_loss),
            "improved": bool(improved),
            "epochs_without_improvement": int(epochs_without_improvement),
        }

        history.append(history_row)

        print(f"Train loss: {train_loss:.6f}")
        print(f"Validation loss: {validation_loss:.6f}")
        print(f"Validation subset accuracy: {validation_metrics['subset_accuracy']:.6f}")
        print(f"Validation hamming accuracy: {validation_metrics['hamming_accuracy']:.6f}")
        print(f"Validation micro-F1: {validation_metrics['micro_f1']:.6f}")
        print(f"Validation macro-F1: {validation_metrics['macro_f1']:.6f}")
        print("Improved:", improved)
        print("Epochs without improvement:", epochs_without_improvement)

        current_checkpoint = make_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            current_val_loss=validation_loss,
            current_val_macro_f1=validation_metrics["macro_f1"],
            current_val_micro_f1=validation_metrics["micro_f1"],
            pos_weight=pos_weight,
            history=history,
            current_metrics=validation_metrics,
            train_summary=train_summary,
            validation_summary=validation_summary,
            training_completed=False,
            early_stopped=False,
        )

        if improved:
            save_checkpoint(BEST_CHECKPOINT_PATH, current_checkpoint)
            print("Best checkpoint saved:", BEST_CHECKPOINT_PATH)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            print(
                f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} "
                "epochs without validation-loss improvement."
            )
            break

    training_completed = True
    elapsed_seconds = time.time() - start_time

    if not history:
        raise RuntimeError("Training history is empty.")

    last_epoch = history[-1]["epoch"]
    last_val_loss = history[-1]["validation_loss"]
    last_val_macro_f1 = history[-1]["validation_macro_f1"]
    last_val_micro_f1 = history[-1]["validation_micro_f1"]

    _, last_metrics, _, _, _ = validate_multilabel_one_epoch(
        model=model,
        dataloader=validation_loader,
        criterion=criterion,
        device=DEVICE,
        threshold=THRESHOLD,
        target_classes=TARGET_CLASSES,
        epoch=None,
    )

    last_checkpoint = make_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=last_epoch,
        best_val_loss=best_val_loss,
        current_val_loss=last_val_loss,
        current_val_macro_f1=last_val_macro_f1,
        current_val_micro_f1=last_val_micro_f1,
        pos_weight=pos_weight,
        history=history,
        current_metrics=last_metrics,
        train_summary=train_summary,
        validation_summary=validation_summary,
        training_completed=training_completed,
        early_stopped=early_stopped,
    )

    save_checkpoint(LAST_CHECKPOINT_PATH, last_checkpoint)

    if not BEST_CHECKPOINT_PATH.exists():
        save_checkpoint(BEST_CHECKPOINT_PATH, last_checkpoint)

    # -----------------------------------------------------
    # Save History / Curves / Summary / Report
    # -----------------------------------------------------

    save_history_csv(history, HISTORY_CSV_PATH)

    plot_multilabel_training_curves(
        history=history,
        output_path=CURVES_PATH,
        title="Multi-Label ResNet18 FC Training Curves",
    )

    if best_metrics is None:
        best_metrics = last_metrics

    if training_completed:
        recommended_next_step = (
            "Proceed to Multi-Label Prompt 4 - fine-tune layer4 and compare with "
            "the fc-only multi-label baseline using validation metrics only."
        )
    else:
        recommended_next_step = (
            "Training did not complete properly. Review logs and reduce learning rate if needed."
        )

    summary = {
        "project_root": str(PROJECT_ROOT),
        "device": str(DEVICE),
        "model_name": MODEL_NAME,
        "training_strategy": TRAINING_STRATEGY,
        "records_csv": str(MULTILABEL_TRAIN_VAL_CSV),
        "target_classes": TARGET_CLASSES,
        "target_columns": TARGET_COLUMNS,
        "threshold": float(THRESHOLD),
        "epochs_requested": int(EPOCHS),
        "epochs_completed": int(len(history)),
        "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
        "early_stopped": bool(early_stopped),
        "learning_rate": float(LEARNING_RATE),
        "weight_decay": float(WEIGHT_DECAY),
        "batch_size": int(BATCH_SIZE),
        "image_size": int(IMAGE_SIZE),
        "parameter_counts": parameter_counts,
        "train_size": int(len(train_dataset)),
        "validation_size": int(len(validation_dataset)),
        "train_positive_counts": train_dataset.get_positive_counts(),
        "validation_positive_counts": validation_dataset.get_positive_counts(),
        "train_label_cardinality": float(train_dataset.get_label_cardinality()),
        "validation_label_cardinality": float(validation_dataset.get_label_cardinality()),
        "pos_weight": [
            float(x)
            for x in pos_weight.detach().cpu().tolist()
        ],
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_validation_subset_accuracy": float(best_metrics["subset_accuracy"]),
        "best_validation_hamming_accuracy": float(best_metrics["hamming_accuracy"]),
        "best_validation_micro_precision": float(best_metrics["micro_precision"]),
        "best_validation_micro_recall": float(best_metrics["micro_recall"]),
        "best_validation_micro_f1": float(best_metrics["micro_f1"]),
        "best_validation_macro_precision": float(best_metrics["macro_precision"]),
        "best_validation_macro_recall": float(best_metrics["macro_recall"]),
        "best_validation_macro_f1": float(best_metrics["macro_f1"]),
        "best_validation_weighted_precision": float(best_metrics["weighted_precision"]),
        "best_validation_weighted_recall": float(best_metrics["weighted_recall"]),
        "best_validation_weighted_f1": float(best_metrics["weighted_f1"]),
        "best_validation_per_class_metrics": best_metrics["per_class_metrics"],
        "last_epoch": int(last_epoch),
        "last_val_loss": float(last_val_loss),
        "last_val_macro_f1": float(last_val_macro_f1),
        "last_val_micro_f1": float(last_val_micro_f1),
        "elapsed_seconds": float(elapsed_seconds),
        "training_completed": bool(training_completed),
        "test_set_used": False,
        "official_single_label_model_untouched": True,
        "history": history,
        "output_files": {
            "best_checkpoint": str(BEST_CHECKPOINT_PATH),
            "last_checkpoint": str(LAST_CHECKPOINT_PATH),
            "history_csv": str(HISTORY_CSV_PATH),
            "summary_json": str(SUMMARY_JSON_PATH),
            "report_md": str(REPORT_MD_PATH),
            "training_curves": str(CURVES_PATH),
        },
        "recommended_next_step": recommended_next_step,
    }

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    save_markdown_report(summary)

    # -----------------------------------------------------
    # Console Summary
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("MULTI-LABEL RESNET18 FC TRAINING SUMMARY")
    print("=" * 80)

    print("Epochs completed:", len(history))
    print("Early stopped:", early_stopped)
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)
    print("Best validation subset accuracy:", best_metrics["subset_accuracy"])
    print("Best validation hamming accuracy:", best_metrics["hamming_accuracy"])
    print("Best validation micro-F1:", best_metrics["micro_f1"])
    print("Best validation macro-F1:", best_metrics["macro_f1"])
    print("Best validation weighted-F1:", best_metrics["weighted_f1"])

    print("\nPer-class validation metrics:")
    for class_name, metrics in best_metrics["per_class_metrics"].items():
        print(class_name, metrics)

    print("\nSaved output files:")
    print("-", BEST_CHECKPOINT_PATH)
    print("-", LAST_CHECKPOINT_PATH)
    print("-", HISTORY_CSV_PATH)
    print("-", SUMMARY_JSON_PATH)
    print("-", REPORT_MD_PATH)
    print("-", CURVES_PATH)

    print("\nFinal recommendation:")
    print(recommended_next_step)

    print("\nMULTI-LABEL RESNET18 FC TRAINING COMPLETED.")


if __name__ == "__main__":
    main()
