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
    unfreeze_resnet_layer4_for_multilabel,
    count_model_parameters,
)
from multilabel_train_utils import (
    compute_pos_weight_from_dataset,
    train_multilabel_one_epoch,
    validate_multilabel_one_epoch,
    save_checkpoint,
    load_checkpoint,
    plot_multilabel_training_curves,
    save_history_csv,
)


# =========================================================
# Configuration
# =========================================================

MODEL_NAME = "multilabel_resnet18_layer4"
TRAINING_STRATEGY = "resnet18_multilabel_layer4_finetuning"

MULTILABEL_TRAIN_VAL_CSV = PROCESSED_RECORDS_DIR / "multilabel_train_val_records.csv"

START_CHECKPOINT_PATH = MODELS_DIR / "multilabel_resnet18_fc_best.pth"

TARGET_CLASSES = ["plastic", "foam", "metal", "other_debris"]

TARGET_COLUMNS = [
    "multilabel_plastic",
    "multilabel_foam",
    "multilabel_metal",
    "multilabel_other_debris",
]

THRESHOLD = 0.5

EPOCHS = 10
EARLY_STOPPING_PATIENCE = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-4

BEST_CHECKPOINT_PATH = MODELS_DIR / "multilabel_resnet18_layer4_best.pth"
LAST_CHECKPOINT_PATH = MODELS_DIR / "multilabel_resnet18_layer4_last.pth"

HISTORY_CSV_PATH = REPORTS_DIR / "multilabel_resnet18_layer4_training_history.csv"
SUMMARY_JSON_PATH = REPORTS_DIR / "multilabel_resnet18_layer4_training_summary.json"
REPORT_MD_PATH = REPORTS_DIR / "multilabel_resnet18_layer4_training_report.md"
COMPARISON_CSV_PATH = REPORTS_DIR / "multilabel_resnet18_fc_vs_layer4_comparison.csv"
CURVES_PATH = PLOTS_DIR / "multilabel_resnet18_layer4_training_curves.png"


# =========================================================
# FC-Only Baseline Reference
# =========================================================

FC_BASELINE = {
    "model_name": "multilabel_resnet18_fc",
    "checkpoint": str(START_CHECKPOINT_PATH),
    "best_val_loss": 0.25249718335168114,
    "best_micro_f1": 0.8036951501154734,
    "best_macro_f1": 0.7965424525974774,
    "best_weighted_f1": 0.8049741138001932,
    "best_hamming_accuracy": 0.7557471264367817,
    "best_subset_accuracy": 0.3793103448275862,
}


# =========================================================
# Helper Functions
# =========================================================

def set_seed(seed: int):
    """Set reproducibility seeds."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def markdown_table_from_dict(title: str, data: dict) -> str:
    """Create a Markdown table from a dictionary."""
    lines = [
        f"### {title}",
        "",
        "| Item | Value |",
        "|---|---:|",
    ]

    for key, value in data.items():
        lines.append(f"| {key} | {value} |")

    return "\n".join(lines)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
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
            text = str(row[col]).replace("\n", " ").replace("|", "/")
            values.append(text)

        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def per_class_metrics_to_dataframe(per_class_metrics: dict) -> pd.DataFrame:
    """Convert per-class metrics dictionary to DataFrame."""
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
        "start_checkpoint": str(START_CHECKPOINT_PATH),
        "layer4_unfrozen": True,
    }

    return checkpoint


def build_comparison_dataframe(layer4_summary: dict) -> pd.DataFrame:
    """Create comparison table between FC baseline and layer4 fine-tuned model."""
    rows = [
        {
            "model": "multilabel_resnet18_fc",
            "checkpoint": str(START_CHECKPOINT_PATH),
            "validation_loss": FC_BASELINE["best_val_loss"],
            "micro_f1": FC_BASELINE["best_micro_f1"],
            "macro_f1": FC_BASELINE["best_macro_f1"],
            "weighted_f1": FC_BASELINE["best_weighted_f1"],
            "hamming_accuracy": FC_BASELINE["best_hamming_accuracy"],
            "subset_accuracy": FC_BASELINE["best_subset_accuracy"],
            "test_set_used": False,
        },
        {
            "model": MODEL_NAME,
            "checkpoint": str(BEST_CHECKPOINT_PATH),
            "validation_loss": layer4_summary["best_val_loss"],
            "micro_f1": layer4_summary["best_validation_micro_f1"],
            "macro_f1": layer4_summary["best_validation_macro_f1"],
            "weighted_f1": layer4_summary["best_validation_weighted_f1"],
            "hamming_accuracy": layer4_summary["best_validation_hamming_accuracy"],
            "subset_accuracy": layer4_summary["best_validation_subset_accuracy"],
            "test_set_used": False,
        },
    ]

    return pd.DataFrame(rows)


def decide_recommendation(summary: dict) -> tuple[str, str]:
    """
    Decide final validation-only recommendation.
    """
    layer4_loss = summary["best_val_loss"]
    layer4_macro_f1 = summary["best_validation_macro_f1"]
    layer4_micro_f1 = summary["best_validation_micro_f1"]

    loss_improved = layer4_loss < FC_BASELINE["best_val_loss"]
    macro_f1_improved = layer4_macro_f1 > FC_BASELINE["best_macro_f1"]
    micro_f1_improved = layer4_micro_f1 > FC_BASELINE["best_micro_f1"]

    if loss_improved and macro_f1_improved:
        status = "layer4_clear_candidate"
        recommendation = (
            "Layer4 fine-tuning improved both validation loss and validation macro-F1 "
            "over the FC-only multi-label baseline. Recommend using "
            "multilabel_resnet18_layer4_best.pth as the current multi-label candidate model. "
            "Do not evaluate the test set yet unless this is a final selected multi-label model."
        )

    elif macro_f1_improved and not loss_improved:
        status = "layer4_promising_but_not_clean_replacement"
        recommendation = (
            "Layer4 fine-tuning improved validation macro-F1 but did not improve validation loss. "
            "Mark it as promising, but not a clean replacement for the FC-only baseline. "
            "Consider validation-only threshold tuning or a lower learning rate before final selection."
        )

    elif micro_f1_improved and not loss_improved:
        status = "layer4_micro_f1_improved_only"
        recommendation = (
            "Layer4 fine-tuning improved validation micro-F1 only, but not validation loss. "
            "Keep the FC-only baseline as the safer current multi-label candidate unless further "
            "validation-only experiments justify replacement."
        )

    else:
        status = "keep_fc_baseline"
        recommendation = (
            "Layer4 fine-tuning did not clearly improve over the FC-only multi-label baseline. "
            "Keep outputs/models/multilabel_resnet18_fc_best.pth as the current multi-label candidate."
        )

    return status, recommendation


def save_markdown_report(summary: dict, comparison_df: pd.DataFrame):
    """Save Markdown report."""
    best_per_class_df = per_class_metrics_to_dataframe(
        summary["best_validation_per_class_metrics"]
    )

    lines = []

    lines.append("# Multi-Label ResNet18 Layer4 Fine-Tuning Report")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This report documents controlled layer4 fine-tuning for the experimental "
        "multi-label ResNet18 branch. The model starts from the FC-only multi-label "
        "baseline checkpoint and unfreezes only layer4 and the final classifier."
    )

    lines.append("")
    lines.append("## 2. Important Notes")
    lines.append("")
    lines.append("- The official single-label model is not overwritten.")
    lines.append("- The FC-only multi-label baseline checkpoint is not overwritten.")
    lines.append("- The test split is not loaded or evaluated.")
    lines.append("- Threshold is fixed at `0.5`.")
    lines.append("- Model selection is based on validation metrics only.")

    lines.append("")
    lines.append("## 3. Starting Checkpoint")
    lines.append("")
    lines.append(f"`{START_CHECKPOINT_PATH}`")

    lines.append("")
    lines.append("## 4. Dataset Summary")
    lines.append("")
    lines.append(markdown_table_from_dict("Train Positive Counts", summary["train_positive_counts"]))
    lines.append("")
    lines.append(markdown_table_from_dict("Validation Positive Counts", summary["validation_positive_counts"]))

    lines.append("")
    lines.append("## 5. Positive Weights")
    lines.append("")
    pos_weight_dict = {
        class_name: summary["pos_weight"][index]
        for index, class_name in enumerate(TARGET_CLASSES)
    }
    lines.append(markdown_table_from_dict("BCE pos_weight", pos_weight_dict))

    lines.append("")
    lines.append("## 6. Model / Training Settings")
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
        "Layer4 unfrozen": True,
        "Device": summary["device"],
    }
    lines.append(markdown_table_from_dict("Settings", settings))

    lines.append("")
    lines.append("## 7. Best Validation Results")
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
    lines.append("## 8. Per-Class Validation Metrics")
    lines.append("")
    lines.append(dataframe_to_markdown(best_per_class_df))

    lines.append("")
    lines.append("## 9. Comparison Against FC-Only Baseline")
    lines.append("")
    lines.append(dataframe_to_markdown(comparison_df))

    lines.append("")
    lines.append("## 10. Important Limitations")
    lines.append("")
    lines.append("- This model has not been evaluated on the held-out test split.")
    lines.append("- Threshold is fixed at 0.5 and may not be optimal for all classes.")
    lines.append("- Multi-label targets are derived from mapped dataset annotations.")
    lines.append("- This model does not replace the official single-label result.")
    lines.append("- Test results must not be used for threshold tuning or model selection.")

    lines.append("")
    lines.append("## 11. Output Files")
    lines.append("")
    for name, path in summary["output_files"].items():
        lines.append(f"- {name}: `{path}`")

    lines.append("")
    lines.append("## 12. Recommendation")
    lines.append("")
    lines.append(summary["recommendation_status"])
    lines.append("")
    lines.append(summary["recommended_next_step"])

    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# Main
# =========================================================

def main():
    set_seed(RANDOM_SEED)

    print("=" * 80)
    print("MULTI-LABEL RESNET18 LAYER4 FINE-TUNING STARTED")
    print("=" * 80)

    print("Project root:", PROJECT_ROOT)
    print("Device:", DEVICE)
    print("Records CSV:", MULTILABEL_TRAIN_VAL_CSV)
    print("Start checkpoint:", START_CHECKPOINT_PATH)
    print("Target classes:", TARGET_CLASSES)
    print("Target columns:", TARGET_COLUMNS)
    print("Threshold:", THRESHOLD)
    print("Important: test split is not loaded.")

    if not MULTILABEL_TRAIN_VAL_CSV.exists():
        raise FileNotFoundError(
            f"Multi-label train/validation CSV not found: {MULTILABEL_TRAIN_VAL_CSV}"
        )

    if not START_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Starting FC-only multi-label checkpoint not found: {START_CHECKPOINT_PATH}"
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

    # -----------------------------------------------------
    # Model Loading
    # -----------------------------------------------------

    model = create_resnet18_multilabel_model(
        num_classes=len(TARGET_CLASSES),
        freeze_backbone=True,
    )

    checkpoint = load_checkpoint(
        path=START_CHECKPOINT_PATH,
        device=DEVICE,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError("Starting checkpoint does not contain 'model_state_dict'.")

    model.load_state_dict(checkpoint["model_state_dict"])

    parameter_counts_before = count_model_parameters(model)

    model = unfreeze_resnet_layer4_for_multilabel(model)

    parameter_counts_after = count_model_parameters(model)

    model = model.to(DEVICE)

    print("\nModel parameter counts:")
    print("Before unfreezing:", parameter_counts_before)
    print("After unfreezing:", parameter_counts_after)

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

    save_history_csv(history, HISTORY_CSV_PATH)

    plot_multilabel_training_curves(
        history=history,
        output_path=CURVES_PATH,
        title="Multi-Label ResNet18 Layer4 Fine-Tuning Curves",
    )

    if best_metrics is None:
        best_metrics = last_metrics

    # -----------------------------------------------------
    # Summary / Comparison / Report
    # -----------------------------------------------------

    temporary_summary = {
        "best_val_loss": float(best_val_loss),
        "best_validation_micro_f1": float(best_metrics["micro_f1"]),
        "best_validation_macro_f1": float(best_metrics["macro_f1"]),
        "best_validation_weighted_f1": float(best_metrics["weighted_f1"]),
        "best_validation_hamming_accuracy": float(best_metrics["hamming_accuracy"]),
        "best_validation_subset_accuracy": float(best_metrics["subset_accuracy"]),
    }

    recommendation_status, recommended_next_step = decide_recommendation(temporary_summary)

    summary = {
        "project_root": str(PROJECT_ROOT),
        "device": str(DEVICE),
        "model_name": MODEL_NAME,
        "training_strategy": TRAINING_STRATEGY,
        "start_checkpoint": str(START_CHECKPOINT_PATH),
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
        "parameter_counts_before_unfreezing": parameter_counts_before,
        "parameter_counts_after_unfreezing": parameter_counts_after,
        "layer4_unfrozen": True,
        "fc_layer_trainable": True,
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
        "fc_baseline_reference": FC_BASELINE,
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
        "fc_only_multilabel_checkpoint_untouched": True,
        "recommendation_status": recommendation_status,
        "recommended_next_step": recommended_next_step,
        "history": history,
        "output_files": {
            "best_checkpoint": str(BEST_CHECKPOINT_PATH),
            "last_checkpoint": str(LAST_CHECKPOINT_PATH),
            "history_csv": str(HISTORY_CSV_PATH),
            "summary_json": str(SUMMARY_JSON_PATH),
            "report_md": str(REPORT_MD_PATH),
            "comparison_csv": str(COMPARISON_CSV_PATH),
            "training_curves": str(CURVES_PATH),
        },
    }

    comparison_df = build_comparison_dataframe(summary)
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False)

    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    save_markdown_report(summary, comparison_df)

    # -----------------------------------------------------
    # Console Summary
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("MULTI-LABEL RESNET18 LAYER4 FINE-TUNING SUMMARY")
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

    print("\nComparison against FC-only baseline:")
    print(comparison_df.to_string(index=False))

    print("\nPer-class validation metrics:")
    for class_name, metrics in best_metrics["per_class_metrics"].items():
        print(class_name, metrics)

    print("\nSaved output files:")
    print("-", BEST_CHECKPOINT_PATH)
    print("-", LAST_CHECKPOINT_PATH)
    print("-", HISTORY_CSV_PATH)
    print("-", SUMMARY_JSON_PATH)
    print("-", REPORT_MD_PATH)
    print("-", COMPARISON_CSV_PATH)
    print("-", CURVES_PATH)

    print("\nFinal recommendation:")
    print(recommendation_status)
    print(recommended_next_step)

    print("\nMULTI-LABEL RESNET18 LAYER4 FINE-TUNING COMPLETED.")


if __name__ == "__main__":
    main()
