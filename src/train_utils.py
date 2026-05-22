from pathlib import Path
import json
import random

import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# Optional tqdm Support
# =========================================================

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# =========================================================
# Reproducibility
# =========================================================

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducible behavior.
    """

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# Basic Metrics
# =========================================================

def calculate_accuracy(outputs, labels):
    """
    Calculate classification accuracy for one batch.
    """

    _, predictions = torch.max(outputs, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)

    if total == 0:
        return 0.0

    return correct / total


def count_trainable_parameters(model):
    """
    Count total and trainable parameters.
    """

    total_params = sum(param.numel() for param in model.parameters())

    trainable_params = sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


# =========================================================
# Class Weights
# =========================================================

def get_class_weights_from_dataset(dataset, class_to_idx, device):
    """
    Compute inverse-frequency class weights from the training dataset.

    Formula:
        weight_c = total_samples / (num_classes * class_count_c)
    """

    class_distribution = dataset.get_class_distribution()

    total_samples = sum(class_distribution.values())
    num_classes = len(class_to_idx)

    if total_samples == 0:
        raise ValueError("Cannot compute class weights because dataset has zero samples.")

    weights = [0.0] * num_classes

    for class_name, class_idx in class_to_idx.items():
        class_count = int(class_distribution.get(class_name, 0))

        if class_count <= 0:
            raise ValueError(
                f"Class '{class_name}' has zero samples. "
                "Cannot compute class weights safely."
            )

        weight = total_samples / (num_classes * class_count)
        weights[class_idx] = float(weight)

    return torch.tensor(weights, dtype=torch.float32, device=device)


# =========================================================
# Training / Validation Loops
# =========================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch=None,
):
    """
    Train model for one epoch.

    Returns:
        average loss and average accuracy.
    """

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    iterator = dataloader

    if tqdm is not None:
        description = "Training" if epoch is None else f"Training epoch {epoch}"
        iterator = tqdm(dataloader, desc=description, leave=False)

    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        _, predictions = torch.max(outputs, dim=1)
        running_correct += (predictions == labels).sum().item()
        running_total += batch_size

    average_loss = running_loss / max(running_total, 1)
    average_accuracy = running_correct / max(running_total, 1)

    return average_loss, average_accuracy


@torch.no_grad()
def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    epoch=None,
):
    """
    Validate model for one epoch.

    Returns:
        average validation loss and average validation accuracy.
    """

    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    iterator = dataloader

    if tqdm is not None:
        description = "Validation" if epoch is None else f"Validation epoch {epoch}"
        iterator = tqdm(dataloader, desc=description, leave=False)

    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size

        _, predictions = torch.max(outputs, dim=1)
        running_correct += (predictions == labels).sum().item()
        running_total += batch_size

    average_loss = running_loss / max(running_total, 1)
    average_accuracy = running_correct / max(running_total, 1)

    return average_loss, average_accuracy


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """
    Collect predictions and true labels from a dataloader.
    """

    model.eval()

    all_true_labels = []
    all_predicted_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predictions = torch.max(outputs, dim=1)

        all_true_labels.extend(labels.detach().cpu().tolist())
        all_predicted_labels.extend(predictions.detach().cpu().tolist())

    return all_true_labels, all_predicted_labels


# =========================================================
# Validation Metric Helpers
# =========================================================

def compute_confusion_matrix_from_predictions(true_labels, predicted_labels, num_classes):
    """
    Compute confusion matrix.

    Rows = true class.
    Columns = predicted class.
    """

    confusion = [
        [0 for _ in range(num_classes)]
        for _ in range(num_classes)
    ]

    for true_label, predicted_label in zip(true_labels, predicted_labels):
        true_label = int(true_label)
        predicted_label = int(predicted_label)

        if 0 <= true_label < num_classes and 0 <= predicted_label < num_classes:
            confusion[true_label][predicted_label] += 1

    return confusion


def compute_per_class_precision_recall_f1(confusion_matrix, idx_to_class):
    """
    Compute per-class precision, recall, and F1-score from confusion matrix.
    """

    num_classes = len(confusion_matrix)
    per_class_metrics = {}

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

        class_name = idx_to_class[class_idx]

        per_class_metrics[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

    return per_class_metrics


def compute_macro_f1(per_class_metrics):
    """
    Compute macro-F1 from per-class metrics.
    """

    if not per_class_metrics:
        return 0.0

    f1_values = [
        metrics["f1"]
        for metrics in per_class_metrics.values()
    ]

    if not f1_values:
        return 0.0

    return float(sum(f1_values) / len(f1_values))


# =========================================================
# Checkpoint Utilities
# =========================================================

def save_checkpoint(path, checkpoint):
    """
    Save training checkpoint.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path, device):
    """
    Load checkpoint.

    weights_only=False is used when supported because our checkpoints store
    model weights plus metadata such as class mappings and history.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# =========================================================
# History Saving / Plotting
# =========================================================

def save_training_history(history, csv_path, json_path=None, extra_summary=None):
    """
    Save training history as CSV and optionally JSON.
    """

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(csv_path, index=False)

    if json_path is not None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "history": history,
        }

        if extra_summary is not None:
            data["summary"] = extra_summary

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


def plot_training_curves(history, output_path, title="Training Curves"):
    """
    Save training and validation curves.

    If validation_macro_f1 exists in history, also plot macro-F1.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        raise ValueError("Cannot plot training curves because history is empty.")

    epochs = [item["epoch"] for item in history]

    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["validation_loss"] for item in history]

    train_accuracies = [item["train_accuracy"] for item in history]
    val_accuracies = [item["validation_accuracy"] for item in history]

    has_macro_f1 = "validation_macro_f1" in history[0]

    if has_macro_f1:
        val_macro_f1 = [item["validation_macro_f1"] for item in history]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_losses, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_losses, marker="o", label="Validation Loss")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_accuracies, marker="o", label="Train Accuracy")
    axes[1].plot(epochs, val_accuracies, marker="o", label="Validation Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    if has_macro_f1:
        axes[2].plot(epochs, val_macro_f1, marker="o", label="Validation Macro-F1")
        axes[2].set_title("Validation Macro-F1")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Macro-F1")
        axes[2].legend()
        axes[2].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)