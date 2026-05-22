import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    PROJECT_ROOT,
    FINAL_SPLIT_CSV,
    REPORTS_DIR,
    PLOTS_DIR,
    MODELS_DIR,
    TARGET_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEVICE,
)

from transforms import get_train_transforms, get_eval_transforms
from dataset import MarineDebrisDataset
from model import create_resnet18_model, unfreeze_resnet_layer4
from train_utils import (
    set_seed,
    get_class_weights_from_dataset,
    train_one_epoch,
    validate_one_epoch,
    collect_predictions,
    compute_confusion_matrix_from_predictions,
    compute_per_class_precision_recall_f1,
    compute_macro_f1,
    save_checkpoint,
    load_checkpoint,
    save_training_history,
    plot_training_curves,
    count_trainable_parameters,
)


# =========================================================
# Fine-Tuning Configuration
# =========================================================

MODEL_NAME = "resnet18_finetuned_layer4"
FINE_TUNING_STRATEGY = "load_baseline_best_unfreeze_layer4_and_fc"

BASELINE_CHECKPOINT_PATH = MODELS_DIR / "resnet18_baseline_best.pth"

BASELINE_BEST_VAL_LOSS = 0.950632022715163
BASELINE_BEST_VAL_ACCURACY = 0.5287356321839081
BASELINE_BEST_EPOCH = 11

EPOCHS = 10
EARLY_STOPPING_PATIENCE = 4

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

BEST_CHECKPOINT_PATH = MODELS_DIR / "resnet18_finetuned_layer4_best.pth"
LAST_CHECKPOINT_PATH = MODELS_DIR / "resnet18_finetuned_layer4_last.pth"

TRAINING_HISTORY_CSV_PATH = REPORTS_DIR / "resnet18_finetuned_layer4_training_history.csv"
TRAINING_SUMMARY_JSON_PATH = REPORTS_DIR / "resnet18_finetuned_layer4_training_summary.json"
TRAINING_CURVES_PATH = PLOTS_DIR / "resnet18_finetuned_layer4_training_curves.png"
TRAINING_REPORT_PATH = REPORTS_DIR / "resnet18_finetuned_layer4_training_report.md"

COMPARISON_JSON_PATH = REPORTS_DIR / "baseline_vs_finetuned_validation_comparison.json"
COMPARISON_MD_PATH = REPORTS_DIR / "baseline_vs_finetuned_validation_comparison.md"


# =========================================================
# Helper Functions
# =========================================================

def markdown_table_from_dict(title, data):
    """
    Create Markdown table from dictionary.
    """

    lines = [
        f"### {title}",
        "",
        "| Item | Value |",
        "|---|---:|",
    ]

    for key, value in data.items():
        lines.append(f"| {key} | {value} |")

    return "\n".join(lines)


def make_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_loss,
    current_val_loss,
    current_val_accuracy,
    current_val_macro_f1,
    history,
    class_weights,
    train_distribution,
    validation_distribution,
    validation_confusion_matrix,
    validation_per_class_metrics,
    training_completed,
    early_stopped,
):
    """
    Create fine-tuning checkpoint dictionary.
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "current_val_loss": float(current_val_loss),
        "current_val_accuracy": float(current_val_accuracy),
        "current_val_macro_f1": float(current_val_macro_f1),
        "class_to_idx": dict(CLASS_TO_IDX),
        "idx_to_class": {
            int(idx): class_name
            for idx, class_name in IDX_TO_CLASS.items()
        },
        "target_classes": list(TARGET_CLASSES),
        "model_name": MODEL_NAME,
        "fine_tuning_strategy": FINE_TUNING_STRATEGY,
        "baseline_checkpoint_path": str(BASELINE_CHECKPOINT_PATH),
        "image_size": int(IMAGE_SIZE),
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "num_classes": int(len(TARGET_CLASSES)),
        "batch_size": int(BATCH_SIZE),
        "learning_rate": float(LEARNING_RATE),
        "weight_decay": float(WEIGHT_DECAY),
        "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
        "history": history,
        "class_weights": [
            float(x)
            for x in class_weights.detach().cpu().tolist()
        ],
        "train_distribution": train_distribution,
        "validation_distribution": validation_distribution,
        "validation_confusion_matrix": validation_confusion_matrix,
        "validation_per_class_metrics": validation_per_class_metrics,
        "training_completed": bool(training_completed),
        "early_stopped": bool(early_stopped),
        "test_set_used": False,
    }

    return checkpoint


def save_markdown_report(summary):
    """
    Save fine-tuning Markdown report.
    """

    history = summary["history"]
    final_history_row = history[-1] if history else {}

    report_lines = []

    report_lines.append("# ResNet18 Layer4 Fine-Tuning Training Report")
    report_lines.append("")
    report_lines.append("## 1. Purpose")
    report_lines.append("")
    report_lines.append(
        "This report documents controlled fine-tuning of ResNet18. "
        "The model starts from the baseline best checkpoint, keeps early layers frozen, "
        "and unfreezes only layer4 and the final classifier. The test set is not used."
    )

    report_lines.append("")
    report_lines.append("## 2. Important Dataset Limitations")
    report_lines.append("")
    report_lines.append("- Dataset is small.")
    report_lines.append("- Class imbalance exists, especially for the `metal` class.")
    report_lines.append("- Many records are derived from ambiguous multi-object images.")
    report_lines.append("- Validation metrics can fluctuate because validation set is small.")
    report_lines.append("- Results here are validation results only, not final test results.")

    report_lines.append("")
    report_lines.append("## 3. Fine-Tuning Configuration")
    report_lines.append("")
    config_rows = {
        "Model": MODEL_NAME,
        "Fine-tuning strategy": FINE_TUNING_STRATEGY,
        "Baseline checkpoint": str(BASELINE_CHECKPOINT_PATH),
        "Epoch limit": EPOCHS,
        "Early stopping patience": EARLY_STOPPING_PATIENCE,
        "Learning rate": LEARNING_RATE,
        "Weight decay": WEIGHT_DECAY,
        "Batch size": BATCH_SIZE,
        "Image size": IMAGE_SIZE,
        "Device": str(DEVICE),
        "Test set used": False,
    }
    report_lines.append(markdown_table_from_dict("Configuration", config_rows))

    report_lines.append("")
    report_lines.append("## 4. Class Mapping")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(CLASS_TO_IDX, indent=4))
    report_lines.append("```")

    report_lines.append("")
    report_lines.append("## 5. Dataset Sizes")
    report_lines.append("")
    size_rows = {
        "Train size": summary["train_size"],
        "Validation size": summary["validation_size"],
    }
    report_lines.append(markdown_table_from_dict("Dataset Sizes", size_rows))

    report_lines.append("")
    report_lines.append("## 6. Train Class Distribution")
    report_lines.append("")
    report_lines.append(markdown_table_from_dict("Train Distribution", summary["train_distribution"]))

    report_lines.append("")
    report_lines.append("## 7. Validation Class Distribution")
    report_lines.append("")
    report_lines.append(markdown_table_from_dict("Validation Distribution", summary["validation_distribution"]))

    report_lines.append("")
    report_lines.append("## 8. Parameter Counts")
    report_lines.append("")
    report_lines.append(markdown_table_from_dict("Before Unfreezing", summary["parameter_counts_before_unfreezing"]))
    report_lines.append("")
    report_lines.append(markdown_table_from_dict("After Unfreezing", summary["parameter_counts_after_unfreezing"]))

    report_lines.append("")
    report_lines.append("## 9. Best Fine-Tuning Validation Result")
    report_lines.append("")
    best_rows = {
        "Best epoch": summary["best_epoch"],
        "Best validation loss": summary["best_val_loss"],
        "Best validation accuracy": summary["best_val_accuracy"],
        "Best validation macro-F1": summary["best_val_macro_f1"],
    }
    report_lines.append(markdown_table_from_dict("Best Validation Result", best_rows))

    report_lines.append("")
    report_lines.append("## 10. Best Validation Per-Class Metrics")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(summary["best_validation_per_class_metrics"], indent=4))
    report_lines.append("```")

    report_lines.append("")
    report_lines.append("## 11. Best Validation Confusion Matrix")
    report_lines.append("")
    report_lines.append("Rows = true labels, columns = predicted labels.")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(summary["best_validation_confusion_matrix"], indent=4))
    report_lines.append("```")

    report_lines.append("")
    report_lines.append("## 12. Final Epoch Logged")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(final_history_row, indent=4))
    report_lines.append("```")

    report_lines.append("")
    report_lines.append("## 13. Baseline Comparison")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(summary["comparison"], indent=4))
    report_lines.append("```")

    report_lines.append("")
    report_lines.append("## 14. Output Files")
    report_lines.append("")
    for name, path in summary["output_files"].items():
        report_lines.append(f"- {name}: `{path}`")

    report_lines.append("")
    report_lines.append("## 15. Final Recommendation")
    report_lines.append("")
    report_lines.append(summary["recommended_next_step"])

    TRAINING_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


def save_comparison_reports(comparison):
    """
    Save baseline-vs-finetuned validation comparison as JSON and Markdown.
    """

    with open(COMPARISON_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)

    lines = []

    lines.append("# Baseline vs Fine-Tuned Validation Comparison")
    lines.append("")
    lines.append("This comparison uses **validation results only**. The test set is not used.")
    lines.append("")
    lines.append("| Metric | Baseline | Fine-Tuned | Improved? |")
    lines.append("|---|---:|---:|---|")
    lines.append(
        f"| Best validation loss | {comparison['baseline_best_val_loss']} | "
        f"{comparison['fine_tuned_best_val_loss']} | "
        f"{comparison['validation_loss_improved_over_baseline']} |"
    )
    lines.append(
        f"| Best validation accuracy | {comparison['baseline_best_val_accuracy']} | "
        f"{comparison['fine_tuned_best_val_accuracy']} | "
        f"{comparison['validation_accuracy_improved_over_baseline']} |"
    )
    lines.append(
        f"| Best validation macro-F1 | Not computed for baseline | "
        f"{comparison['fine_tuned_best_val_macro_f1']} | Reference only |"
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(comparison["final_recommendation"])

    COMPARISON_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# Main Fine-Tuning Script
# =========================================================

def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("RESNET18 LAYER4 FINE-TUNING STARTED")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Device:", DEVICE)
    print("Final split CSV:", FINAL_SPLIT_CSV)
    print("Class mapping:", CLASS_TO_IDX)
    print("Baseline checkpoint:", BASELINE_CHECKPOINT_PATH)

    if not FINAL_SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Final split CSV not found: {FINAL_SPLIT_CSV}"
        )

    if not BASELINE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Baseline best checkpoint not found: {BASELINE_CHECKPOINT_PATH}\n"
            "Run src/train_baseline.py first."
        )

    # -----------------------------------------------------
    # Datasets and DataLoaders
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

    train_distribution = train_dataset.get_class_distribution()
    validation_distribution = validation_dataset.get_class_distribution()

    print("\nDataset sizes:")
    print("Train:", len(train_dataset))
    print("Validation:", len(validation_dataset))

    print("\nTrain class distribution:")
    for class_name, count in train_distribution.items():
        print(f"- {class_name}: {count}")

    print("\nValidation class distribution:")
    for class_name, count in validation_distribution.items():
        print(f"- {class_name}: {count}")

    # -----------------------------------------------------
    # Model Loading and Fine-Tuning Setup
    # -----------------------------------------------------

    model = create_resnet18_model(
        num_classes=len(TARGET_CLASSES),
        freeze_backbone=True,
    )

    baseline_checkpoint = load_checkpoint(
        path=BASELINE_CHECKPOINT_PATH,
        device=DEVICE,
    )

    if "model_state_dict" not in baseline_checkpoint:
        raise KeyError("Baseline checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(baseline_checkpoint["model_state_dict"])

    parameter_counts_before_unfreezing = count_trainable_parameters(model)

    print("\nParameter counts before unfreezing layer4:")
    print("Total parameters:", parameter_counts_before_unfreezing["total_params"])
    print("Trainable parameters:", parameter_counts_before_unfreezing["trainable_params"])

    model = unfreeze_resnet_layer4(model)

    parameter_counts_after_unfreezing = count_trainable_parameters(model)

    print("\nParameter counts after unfreezing layer4:")
    print("Total parameters:", parameter_counts_after_unfreezing["total_params"])
    print("Trainable parameters:", parameter_counts_after_unfreezing["trainable_params"])

    model = model.to(DEVICE)

    class_weights = get_class_weights_from_dataset(
        dataset=train_dataset,
        class_to_idx=CLASS_TO_IDX,
        device=DEVICE,
    )

    print("\nClass weights:")
    for class_name, class_idx in CLASS_TO_IDX.items():
        print(f"- {class_name}: {class_weights[class_idx].item():.6f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Fine-Tuning Loop
    # -----------------------------------------------------

    history = []

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_val_macro_f1 = 0.0
    best_epoch = 0
    best_validation_confusion_matrix = None
    best_validation_per_class_metrics = None

    epochs_without_improvement = 0
    early_stopped = False
    training_completed = False

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "-" * 80)
        print(f"Fine-tuning epoch {epoch}/{EPOCHS}")
        print("-" * 80)

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
            epoch=epoch,
        )

        validation_loss, validation_accuracy = validate_one_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=DEVICE,
            epoch=epoch,
        )

        validation_true_labels, validation_predicted_labels = collect_predictions(
            model=model,
            dataloader=validation_loader,
            device=DEVICE,
        )

        validation_confusion_matrix = compute_confusion_matrix_from_predictions(
            true_labels=validation_true_labels,
            predicted_labels=validation_predicted_labels,
            num_classes=len(TARGET_CLASSES),
        )

        validation_per_class_metrics = compute_per_class_precision_recall_f1(
            confusion_matrix=validation_confusion_matrix,
            idx_to_class=IDX_TO_CLASS,
        )

        validation_macro_f1 = compute_macro_f1(validation_per_class_metrics)

        improved = validation_loss < best_val_loss

        if improved:
            best_val_loss = validation_loss
            best_val_accuracy = validation_accuracy
            best_val_macro_f1 = validation_macro_f1
            best_epoch = epoch
            best_validation_confusion_matrix = validation_confusion_matrix
            best_validation_per_class_metrics = validation_per_class_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "validation_loss": float(validation_loss),
            "validation_accuracy": float(validation_accuracy),
            "validation_macro_f1": float(validation_macro_f1),
            "best_val_loss_so_far": float(best_val_loss),
            "improved": bool(improved),
            "epochs_without_improvement": int(epochs_without_improvement),
        }

        history.append(epoch_record)

        print(f"Train loss: {train_loss:.6f}")
        print(f"Train accuracy: {train_accuracy:.6f}")
        print(f"Validation loss: {validation_loss:.6f}")
        print(f"Validation accuracy: {validation_accuracy:.6f}")
        print(f"Validation macro-F1: {validation_macro_f1:.6f}")
        print("Improved:", improved)
        print("Epochs without improvement:", epochs_without_improvement)

        current_checkpoint = make_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            current_val_loss=validation_loss,
            current_val_accuracy=validation_accuracy,
            current_val_macro_f1=validation_macro_f1,
            history=history,
            class_weights=class_weights,
            train_distribution=train_distribution,
            validation_distribution=validation_distribution,
            validation_confusion_matrix=validation_confusion_matrix,
            validation_per_class_metrics=validation_per_class_metrics,
            training_completed=False,
            early_stopped=False,
        )

        if improved:
            save_checkpoint(BEST_CHECKPOINT_PATH, current_checkpoint)
            print("Best fine-tuned checkpoint saved:", BEST_CHECKPOINT_PATH)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            early_stopped = True
            print(
                f"Early stopping triggered after {EARLY_STOPPING_PATIENCE} "
                "epochs without validation-loss improvement."
            )
            break

    training_completed = True
    elapsed_seconds = time.time() - start_time

    # -----------------------------------------------------
    # Save Last Checkpoint
    # -----------------------------------------------------

    if not history:
        raise RuntimeError("Fine-tuning history is empty. Training did not run.")

    last_epoch = history[-1]["epoch"]
    last_val_loss = history[-1]["validation_loss"]
    last_val_accuracy = history[-1]["validation_accuracy"]
    last_val_macro_f1 = history[-1]["validation_macro_f1"]

    last_true_labels, last_predicted_labels = collect_predictions(
        model=model,
        dataloader=validation_loader,
        device=DEVICE,
    )

    last_confusion_matrix = compute_confusion_matrix_from_predictions(
        true_labels=last_true_labels,
        predicted_labels=last_predicted_labels,
        num_classes=len(TARGET_CLASSES),
    )

    last_per_class_metrics = compute_per_class_precision_recall_f1(
        confusion_matrix=last_confusion_matrix,
        idx_to_class=IDX_TO_CLASS,
    )

    last_checkpoint = make_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=last_epoch,
        best_val_loss=best_val_loss,
        current_val_loss=last_val_loss,
        current_val_accuracy=last_val_accuracy,
        current_val_macro_f1=last_val_macro_f1,
        history=history,
        class_weights=class_weights,
        train_distribution=train_distribution,
        validation_distribution=validation_distribution,
        validation_confusion_matrix=last_confusion_matrix,
        validation_per_class_metrics=last_per_class_metrics,
        training_completed=training_completed,
        early_stopped=early_stopped,
    )

    save_checkpoint(LAST_CHECKPOINT_PATH, last_checkpoint)

    if not BEST_CHECKPOINT_PATH.exists():
        save_checkpoint(BEST_CHECKPOINT_PATH, last_checkpoint)

    # -----------------------------------------------------
    # Baseline vs Fine-Tuned Comparison
    # -----------------------------------------------------

    validation_loss_improved_over_baseline = best_val_loss < BASELINE_BEST_VAL_LOSS
    validation_accuracy_improved_over_baseline = best_val_accuracy > BASELINE_BEST_VAL_ACCURACY

    training_unstable = (
        best_epoch == 0
        or best_val_loss == float("inf")
        or len(history) < 2
    )

    if validation_loss_improved_over_baseline:
        final_recommendation = (
            "Proceed to Prompt 9 - Final Test Evaluation, Confusion Matrix, "
            "Classification Report, and Error Analysis. Use the fine-tuned best checkpoint, "
            "but still do not overclaim because the dataset is small and imbalanced."
        )
    elif validation_accuracy_improved_over_baseline and not training_unstable:
        final_recommendation = (
            "Fine-tuning did not improve validation loss, but validation accuracy improved. "
            "Review baseline and fine-tuned validation reports before choosing which checkpoint "
            "to use for final test evaluation."
        )
    elif training_unstable:
        final_recommendation = (
            "Fine-tuned training appears unstable. Prefer the baseline best checkpoint for final evaluation."
        )
    else:
        final_recommendation = (
            "Fine-tuning did not clearly improve validation loss or accuracy. "
            "Use the baseline best checkpoint for final test evaluation unless manual review suggests otherwise."
        )

    comparison = {
        "baseline_best_epoch": int(BASELINE_BEST_EPOCH),
        "baseline_best_val_loss": float(BASELINE_BEST_VAL_LOSS),
        "baseline_best_val_accuracy": float(BASELINE_BEST_VAL_ACCURACY),
        "fine_tuned_best_epoch": int(best_epoch),
        "fine_tuned_best_val_loss": float(best_val_loss),
        "fine_tuned_best_val_accuracy": float(best_val_accuracy),
        "fine_tuned_best_val_macro_f1": float(best_val_macro_f1),
        "validation_loss_improved_over_baseline": bool(validation_loss_improved_over_baseline),
        "validation_accuracy_improved_over_baseline": bool(validation_accuracy_improved_over_baseline),
        "training_unstable": bool(training_unstable),
        "test_set_used": False,
        "final_recommendation": final_recommendation,
    }

    save_comparison_reports(comparison)

    # -----------------------------------------------------
    # Save History, Curves, Summary, Report
    # -----------------------------------------------------

    summary = {
        "project_root": str(PROJECT_ROOT),
        "device": str(DEVICE),
        "model_name": MODEL_NAME,
        "fine_tuning_strategy": FINE_TUNING_STRATEGY,
        "baseline_checkpoint_path": str(BASELINE_CHECKPOINT_PATH),
        "epochs_requested": int(EPOCHS),
        "epochs_completed": int(len(history)),
        "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
        "early_stopped": bool(early_stopped),
        "learning_rate": float(LEARNING_RATE),
        "weight_decay": float(WEIGHT_DECAY),
        "batch_size": int(BATCH_SIZE),
        "image_size": int(IMAGE_SIZE),
        "target_classes": list(TARGET_CLASSES),
        "class_to_idx": dict(CLASS_TO_IDX),
        "idx_to_class": {
            str(idx): class_name
            for idx, class_name in IDX_TO_CLASS.items()
        },
        "train_size": int(len(train_dataset)),
        "validation_size": int(len(validation_dataset)),
        "train_distribution": train_distribution,
        "validation_distribution": validation_distribution,
        "class_weights": [
            float(x)
            for x in class_weights.detach().cpu().tolist()
        ],
        "parameter_counts_before_unfreezing": parameter_counts_before_unfreezing,
        "parameter_counts_after_unfreezing": parameter_counts_after_unfreezing,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_validation_confusion_matrix": best_validation_confusion_matrix,
        "best_validation_per_class_metrics": best_validation_per_class_metrics,
        "last_epoch": int(last_epoch),
        "last_val_loss": float(last_val_loss),
        "last_val_accuracy": float(last_val_accuracy),
        "last_val_macro_f1": float(last_val_macro_f1),
        "elapsed_seconds": float(elapsed_seconds),
        "history": history,
        "comparison": comparison,
        "test_set_used": False,
        "training_completed": bool(training_completed),
        "recommended_next_step": final_recommendation,
        "output_files": {
            "best_checkpoint": str(BEST_CHECKPOINT_PATH),
            "last_checkpoint": str(LAST_CHECKPOINT_PATH),
            "training_history_csv": str(TRAINING_HISTORY_CSV_PATH),
            "training_summary_json": str(TRAINING_SUMMARY_JSON_PATH),
            "training_curves": str(TRAINING_CURVES_PATH),
            "training_report": str(TRAINING_REPORT_PATH),
            "comparison_json": str(COMPARISON_JSON_PATH),
            "comparison_markdown": str(COMPARISON_MD_PATH),
        },
    }

    save_training_history(
        history=history,
        csv_path=TRAINING_HISTORY_CSV_PATH,
        json_path=TRAINING_SUMMARY_JSON_PATH,
        extra_summary=summary,
    )

    plot_training_curves(
        history=history,
        output_path=TRAINING_CURVES_PATH,
        title="ResNet18 Layer4 Fine-Tuning Curves",
    )

    save_markdown_report(summary)

    # -----------------------------------------------------
    # Console Summary
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("RESNET18 LAYER4 FINE-TUNING SUMMARY")
    print("=" * 80)

    print("\n1. Dataset")
    print("Train size:", len(train_dataset))
    print("Validation size:", len(validation_dataset))
    print("Test set used: False")

    print("\n2. Model")
    print("Model:", MODEL_NAME)
    print("Fine-tuning strategy:", FINE_TUNING_STRATEGY)
    print("Baseline checkpoint:", BASELINE_CHECKPOINT_PATH)
    print("Trainable parameters before unfreezing:", parameter_counts_before_unfreezing["trainable_params"])
    print("Trainable parameters after unfreezing:", parameter_counts_after_unfreezing["trainable_params"])

    print("\n3. Fine-tuning result")
    print("Epochs completed:", len(history))
    print("Early stopped:", early_stopped)
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)
    print("Best validation accuracy:", best_val_accuracy)
    print("Best validation macro-F1:", best_val_macro_f1)
    print("Last validation loss:", last_val_loss)
    print("Last validation accuracy:", last_val_accuracy)
    print("Last validation macro-F1:", last_val_macro_f1)

    print("\n4. Baseline comparison")
    print("Baseline best validation loss:", BASELINE_BEST_VAL_LOSS)
    print("Baseline best validation accuracy:", BASELINE_BEST_VAL_ACCURACY)
    print("Validation loss improved over baseline:", validation_loss_improved_over_baseline)
    print("Validation accuracy improved over baseline:", validation_accuracy_improved_over_baseline)

    print("\n5. Output files")
    print("-", BEST_CHECKPOINT_PATH)
    print("-", LAST_CHECKPOINT_PATH)
    print("-", TRAINING_HISTORY_CSV_PATH)
    print("-", TRAINING_SUMMARY_JSON_PATH)
    print("-", TRAINING_CURVES_PATH)
    print("-", TRAINING_REPORT_PATH)
    print("-", COMPARISON_JSON_PATH)
    print("-", COMPARISON_MD_PATH)

    print("\n6. Final recommendation")
    print(final_recommendation)

    print("\nRESNET18 LAYER4 FINE-TUNING COMPLETED.")


if __name__ == "__main__":
    main()
