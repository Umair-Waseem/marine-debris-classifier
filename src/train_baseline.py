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
from model import create_resnet18_model
from train_utils import (
    set_seed,
    get_class_weights_from_dataset,
    train_one_epoch,
    validate_one_epoch,
    save_checkpoint,
    save_training_history,
    plot_training_curves,
    count_trainable_parameters,
)


# =========================================================
# Baseline Training Configuration
# =========================================================

MODEL_NAME = "resnet18_baseline"
FREEZE_BACKBONE = True

EPOCHS = 15
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

BEST_CHECKPOINT_PATH = MODELS_DIR / "resnet18_baseline_best.pth"
LAST_CHECKPOINT_PATH = MODELS_DIR / "resnet18_baseline_last.pth"

TRAINING_HISTORY_CSV_PATH = REPORTS_DIR / "resnet18_baseline_training_history.csv"
TRAINING_SUMMARY_JSON_PATH = REPORTS_DIR / "resnet18_baseline_training_summary.json"
TRAINING_CURVES_PATH = PLOTS_DIR / "resnet18_baseline_training_curves.png"
TRAINING_REPORT_PATH = REPORTS_DIR / "resnet18_baseline_training_report.md"


# =========================================================
# Helper Functions
# =========================================================

def make_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_loss,
    current_val_loss,
    current_val_accuracy,
    history,
    class_weights,
    train_distribution,
    validation_distribution,
    training_completed,
    early_stopped,
):
    """
    Create checkpoint dictionary with all required metadata.
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "current_val_loss": float(current_val_loss),
        "current_val_accuracy": float(current_val_accuracy),
        "class_to_idx": dict(CLASS_TO_IDX),
        "idx_to_class": {
            int(idx): class_name
            for idx, class_name in IDX_TO_CLASS.items()
        },
        "target_classes": list(TARGET_CLASSES),
        "model_name": MODEL_NAME,
        "freeze_backbone": bool(FREEZE_BACKBONE),
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
        "training_completed": bool(training_completed),
        "early_stopped": bool(early_stopped),
    }

    return checkpoint


def markdown_table_from_dict(title, data):
    """
    Create a Markdown table from a dictionary.
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


def save_markdown_report(summary):
    """
    Save Markdown training report.
    """

    history = summary["history"]
    final_history_row = history[-1] if history else {}

    report_lines = []

    report_lines.append("# ResNet18 Baseline Training Report")
    report_lines.append("")
    report_lines.append("## 1. Purpose")
    report_lines.append("")
    report_lines.append(
        "This report documents the baseline transfer learning training step. "
        "A pretrained ResNet18 model was used with the backbone frozen and only "
        "the final classifier layer trained. The test set was not used."
    )

    report_lines.append("")
    report_lines.append("## 2. Important Dataset Limitations")
    report_lines.append("")
    report_lines.append("- Dataset is small.")
    report_lines.append("- Class imbalance exists, especially for the `metal` class.")
    report_lines.append("- Many labels are derived from multi-object images using a dominant-label conversion rule.")
    report_lines.append("- Results from this step are validation results only, not final test results.")

    report_lines.append("")
    report_lines.append("## 3. Training Configuration")
    report_lines.append("")
    config_rows = {
        "Model": MODEL_NAME,
        "Backbone frozen": FREEZE_BACKBONE,
        "Epoch limit": EPOCHS,
        "Early stopping patience": EARLY_STOPPING_PATIENCE,
        "Learning rate": LEARNING_RATE,
        "Weight decay": WEIGHT_DECAY,
        "Batch size": BATCH_SIZE,
        "Image size": IMAGE_SIZE,
        "Device": str(DEVICE),
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
    report_lines.append("## 8. Class Weights")
    report_lines.append("")
    class_weight_rows = {
        class_name: summary["class_weights"][CLASS_TO_IDX[class_name]]
        for class_name in TARGET_CLASSES
    }
    report_lines.append(markdown_table_from_dict("Class Weights", class_weight_rows))

    report_lines.append("")
    report_lines.append("## 9. Parameter Counts")
    report_lines.append("")
    report_lines.append(markdown_table_from_dict("Parameters", summary["parameter_counts"]))

    report_lines.append("")
    report_lines.append("## 10. Best Validation Result")
    report_lines.append("")
    best_rows = {
        "Best epoch": summary["best_epoch"],
        "Best validation loss": summary["best_val_loss"],
        "Best validation accuracy": summary["best_val_accuracy"],
    }
    report_lines.append(markdown_table_from_dict("Best Validation Result", best_rows))

    report_lines.append("")
    report_lines.append("## 11. Final Epoch Logged")
    report_lines.append("")
    report_lines.append("```json")
    report_lines.append(json.dumps(final_history_row, indent=4))
    report_lines.append("```")

    report_lines.append("")
    report_lines.append("## 12. Output Files")
    report_lines.append("")
    for name, path in summary["output_files"].items():
        report_lines.append(f"- {name}: `{path}`")

    report_lines.append("")
    report_lines.append("## 13. Final Recommendation")
    report_lines.append("")
    report_lines.append(summary["recommended_next_step"])

    TRAINING_REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


# =========================================================
# Main Training Script
# =========================================================

def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("RESNET18 BASELINE TRAINING STARTED")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Device:", DEVICE)
    print("Final split CSV:", FINAL_SPLIT_CSV)
    print("Class mapping:", CLASS_TO_IDX)

    if not FINAL_SPLIT_CSV.exists():
        raise FileNotFoundError(
            f"Final split CSV not found: {FINAL_SPLIT_CSV}\n"
            "Run the dataset pipeline preparation and audit steps first."
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
    # Model, Loss, Optimizer
    # -----------------------------------------------------

    model = create_resnet18_model(
        num_classes=len(TARGET_CLASSES),
        freeze_backbone=FREEZE_BACKBONE,
    )

    model = model.to(DEVICE)

    parameter_counts = count_trainable_parameters(model)

    print("\nParameter counts:")
    print("Total parameters:", parameter_counts["total_params"])
    print("Trainable parameters:", parameter_counts["trainable_params"])

    class_weights = get_class_weights_from_dataset(
        dataset=train_dataset,
        class_to_idx=CLASS_TO_IDX,
        device=DEVICE,
    )

    print("\nClass weights:")
    for class_name, class_idx in CLASS_TO_IDX.items():
        print(f"- {class_name}: {class_weights[class_idx].item():.6f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Training Loop
    # -----------------------------------------------------

    history = []
    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    training_completed = False

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        print("\n" + "-" * 80)
        print(f"Epoch {epoch}/{EPOCHS}")
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

        improved = validation_loss < best_val_loss

        if improved:
            best_val_loss = validation_loss
            best_val_accuracy = validation_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_record = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "validation_loss": float(validation_loss),
            "validation_accuracy": float(validation_accuracy),
            "best_val_loss_so_far": float(best_val_loss),
            "improved": bool(improved),
            "epochs_without_improvement": int(epochs_without_improvement),
        }

        history.append(epoch_record)

        print(f"Train loss: {train_loss:.6f}")
        print(f"Train accuracy: {train_accuracy:.6f}")
        print(f"Validation loss: {validation_loss:.6f}")
        print(f"Validation accuracy: {validation_accuracy:.6f}")
        print("Improved:", improved)
        print("Epochs without improvement:", epochs_without_improvement)

        current_checkpoint = make_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_loss=best_val_loss,
            current_val_loss=validation_loss,
            current_val_accuracy=validation_accuracy,
            history=history,
            class_weights=class_weights,
            train_distribution=train_distribution,
            validation_distribution=validation_distribution,
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

    # -----------------------------------------------------
    # Save Last Checkpoint
    # -----------------------------------------------------

    last_epoch = history[-1]["epoch"] if history else 0
    last_val_loss = history[-1]["validation_loss"] if history else float("inf")
    last_val_accuracy = history[-1]["validation_accuracy"] if history else 0.0

    last_checkpoint = make_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=last_epoch,
        best_val_loss=best_val_loss,
        current_val_loss=last_val_loss,
        current_val_accuracy=last_val_accuracy,
        history=history,
        class_weights=class_weights,
        train_distribution=train_distribution,
        validation_distribution=validation_distribution,
        training_completed=training_completed,
        early_stopped=early_stopped,
    )

    save_checkpoint(LAST_CHECKPOINT_PATH, last_checkpoint)

    # If no improvement ever happened, save last as best fallback.
    if not BEST_CHECKPOINT_PATH.exists():
        save_checkpoint(BEST_CHECKPOINT_PATH, last_checkpoint)

    # -----------------------------------------------------
    # Save History, Curves, Summary, Report
    # -----------------------------------------------------

    validation_improved_at_least_once = best_epoch > 0

    if validation_improved_at_least_once and history:
        recommended_next_step = (
            "Proceed to Prompt 8 - Fine-Tuning Last ResNet Block and Comparing with Baseline. "
            "Do not use the test set yet."
        )
    else:
        recommended_next_step = (
            "Do not fine-tune yet. Debug baseline training because validation loss did not improve."
        )

    summary = {
        "project_root": str(PROJECT_ROOT),
        "device": str(DEVICE),
        "model_name": MODEL_NAME,
        "freeze_backbone": bool(FREEZE_BACKBONE),
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
        "parameter_counts": parameter_counts,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "best_val_accuracy": float(best_val_accuracy),
        "last_epoch": int(last_epoch),
        "last_val_loss": float(last_val_loss),
        "last_val_accuracy": float(last_val_accuracy),
        "elapsed_seconds": float(elapsed_seconds),
        "history": history,
        "test_set_used": False,
        "training_completed": bool(training_completed),
        "validation_improved_at_least_once": bool(validation_improved_at_least_once),
        "recommended_next_step": recommended_next_step,
        "output_files": {
            "best_checkpoint": str(BEST_CHECKPOINT_PATH),
            "last_checkpoint": str(LAST_CHECKPOINT_PATH),
            "training_history_csv": str(TRAINING_HISTORY_CSV_PATH),
            "training_summary_json": str(TRAINING_SUMMARY_JSON_PATH),
            "training_curves": str(TRAINING_CURVES_PATH),
            "training_report": str(TRAINING_REPORT_PATH),
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
    )

    save_markdown_report(summary)

    # -----------------------------------------------------
    # Console Summary
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("RESNET18 BASELINE TRAINING SUMMARY")
    print("=" * 80)

    print("\n1. Dataset")
    print("Train size:", len(train_dataset))
    print("Validation size:", len(validation_dataset))
    print("Test set used: False")

    print("\n2. Model")
    print("Model:", MODEL_NAME)
    print("Backbone frozen:", FREEZE_BACKBONE)
    print("Total parameters:", parameter_counts["total_params"])
    print("Trainable parameters:", parameter_counts["trainable_params"])

    print("\n3. Training result")
    print("Epochs completed:", len(history))
    print("Early stopped:", early_stopped)
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val_loss)
    print("Best validation accuracy:", best_val_accuracy)
    print("Last validation loss:", last_val_loss)
    print("Last validation accuracy:", last_val_accuracy)

    print("\n4. Output files")
    print("-", BEST_CHECKPOINT_PATH)
    print("-", LAST_CHECKPOINT_PATH)
    print("-", TRAINING_HISTORY_CSV_PATH)
    print("-", TRAINING_SUMMARY_JSON_PATH)
    print("-", TRAINING_CURVES_PATH)
    print("-", TRAINING_REPORT_PATH)

    print("\n5. Final recommendation")
    print(recommended_next_step)

    print("\nRESNET18 BASELINE TRAINING COMPLETED.")


if __name__ == "__main__":
    main()
